from junwei.utils.tools import get_ind_complexity, get_ind_dimension_gap
from junwei.env.env_manager import StaticFlexibleEnvManager4Test
from junwei.utils.init import *
import sys
from junwei.utils import save_file

"""
    created by JPang
    Date: 12/9/2023
    Test on given testcase 
"""


def getGPObjMakespan(testEnvManager, inds, toolbox):
    objMakespan = np.full((len(inds), len(testEnvManager.test_envs)), np.inf)
    for ins_count in range(len(testEnvManager.test_envs)):
        testEnvManager.set_test_env(ins_count)
        objective_value_ind = toolbox.map(toolbox.evaluate, inds)
        objMakespan[:, ins_count] = [float(i[0]) for i in objective_value_ind]
    return objMakespan


def getInitializer(env_manager, *args):
    if args[0]:
        pset, toolbox = standardGPInitializer(env_manager, num_objs=args[1])
    else:
        pset, toolbox = standardGPInitializer(env_manager)
    return pset, toolbox


def main(seed_algo, dataSetType, dataSetName):
    load_dict = json.load(open("./gp_paras.json", 'r'))
    test_paras = load_dict["test_paras"]
    algo_name = sys.argv[2]
    algo_paras = test_paras[dataSetType][dataSetName][0]
    num_ins = test_paras[dataSetType][dataSetName][1]
    check_multi_objs = load_dict[algo_name + "_paras"]["multi_objs"]
    num_objs = load_dict[algo_name + "_paras"]["num_objs"] if check_multi_objs else None
    testEnvManager = StaticFlexibleEnvManager4Test(dataSetName, num_ins)
    objectiveValueBestIndAllGenAllParas = np.full((load_dict[sys.argv[2] + "_paras"]["num_gen"], len(algo_paras)), np.inf)
    pset, toolbox = getInitializer(testEnvManager, check_multi_objs, num_objs)
    if not check_multi_objs:
        # single obj
        for algoParaCount, algo_para in enumerate(algo_paras):
            inds = standardGPFromStr2Ind(seed_algo, algo_para, pset)
            objective_value = getGPObjMakespan(testEnvManager, inds, toolbox)
            objectiveValueBestIndAllGenAllParas[:, algoParaCount] = np.mean(objective_value, axis=1)
        save_file.save_test_results(check_multi_objs, objectiveValueBestIndAllGenAllParas)
    else:
        # multi objs
        structure_complexity_dict = load_dict["structure_complexity_dict"]["values"]
        dimension_dict = load_dict["dimension_dict"]["values"]
        toolbox.register("get_structure_complexity", get_ind_complexity,
                         structure_complexity_dict=structure_complexity_dict)
        toolbox.register("get_dimension_gap", get_ind_dimension_gap, dimension_dict=dimension_dict)
        multiObjsMakespan = []
        multiObjsStructureComplexity = []
        multiObjsDimensionGap = []
        for algoParaCount, algo_para in enumerate(algo_paras):
            bestIndAllGen, pfAllGen = multiObjsGPFromStr2Ind(seed_algo, algo_para, pset)
            # for best inds for each generation
            objMakespan = getGPObjMakespan(testEnvManager, bestIndAllGen, toolbox)
            objectiveValueBestIndAllGenAllParas[:, algoParaCount] = np.mean(objMakespan, axis=1)
            # for PF in each generation
            makespan = []
            structure_complexity_list = []
            dimension_gap_list = []
            # for pfEachGen in pfAllGen:
            #     makespan.append(np.mean(getGPObjMakespan(testEnvManager, pfEachGen, toolbox), axis=1))
            #     structure_complexity_list.append(toolbox.multiprocessing(toolbox.get_structure_complexity, pfEachGen))
            #     dimension_gap_list.append(toolbox.multiprocessing(toolbox.get_dimension_gap, pfEachGen))

            # juest for the non-dominated solutions in the last generation
            makespan.append(np.mean(getGPObjMakespan(testEnvManager, pfAllGen[-1], toolbox), axis=1))
            structure_complexity_list.append(toolbox.map(toolbox.get_structure_complexity, pfAllGen[-1]))
            dimension_gap_list.append(toolbox.map(toolbox.get_dimension_gap, pfAllGen[-1]))

            multiObjsMakespan.append(tuple(makespan))
            multiObjsStructureComplexity.append(tuple(structure_complexity_list))
            multiObjsDimensionGap.append(tuple(dimension_gap_list))
        save_file.save_test_results(check_multi_objs, objectiveValueBestIndAllGenAllParas, multiObjsMakespan,
                                    multiObjsStructureComplexity, multiObjsDimensionGap)
