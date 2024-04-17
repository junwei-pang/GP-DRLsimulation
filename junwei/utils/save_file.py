import os
import pandas as pd
import numpy as np
import sys
import json
import datetime

"""
    created by JPang
    to save experimental results of GP for job shop 
    date: 02/08/2023
"""


def save_train_results(pset, toolbox, pop_list, best_ind_all_gen, best_fit_all_gen, train_time):
    algo_name = sys.argv[2]
    seed = int(sys.argv[3])
    num_jobs = int(sys.argv[4])
    num_mas = int(sys.argv[5])
    save_path = './save/Train/{0}/Jobs{1}Mas{2}'.format(algo_name, num_jobs, num_mas)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = '{0}/seed{1}.json'.format(save_path, seed)
    with open("./gp_paras.json", 'r') as load_f:
        load_dict = json.load(load_f)
    algo_paras = load_dict[algo_name + "_paras"]
    pset: list = list(pset.mapping.keys())
    config: dict = {
        "algo_name": algo_name,
        "selection": toolbox.select.func.__name__,
        "cx_operator": toolbox.mate.func.__name__,
        "mut_operator": toolbox.mutate.func.__name__,
        "pset": pset
    }
    config.update(algo_paras)

    if algo_paras["multi_objs"]:
        # for multi objectives
        population: list = [
            {
                "gen": pf_count+1,
                "whole population": pop_list[pf_count],
                "pareto fronts": {
                    "inds": [ind for ind in pfEachGen],
                    "fitness": best_fit_all_gen[pf_count]
                }
            }
            for pf_count, pfEachGen in enumerate(best_ind_all_gen)
        ]
    else:
        # for single objective
        population: list = [
            {
                "gen": ind_count+1,
                "whole population": pop_list[ind_count],
                "best ind in each generation": {
                    "ind": ind,
                    "fitness": best_fit_all_gen[ind_count]
                }
            }
            for ind_count, ind in enumerate(best_ind_all_gen)
        ]

    results: dict = {
        "configuration": config,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "population": population,
        "training time": [float(i) for i in train_time]
    }
    json.dump(results, fp=open(file_path, mode='w'))


"""
    json
    "configuration": dict{
        "algo_name","selection","cx_operator","mut_operator"
        "pset": list
        "num_gen","num_pop","cxpb","mutpb","num_objs","multi_objs"
    }
    "datetime": str
    "effectiveness": dict{
        {"training size: JobsXMasX"}: dict{
            list
                dict{
                   "gen"
                   "objective value" 
                }
                "configuration": dict{
                    "test case"
                    "number of test case"
                }
        }
    }
    "multi objectives": dict{
        {"training size: JobsXMasX"}: dict{
            list
                dict{
                   "gen"
                   "makespan"
                   "structure complexity"
                   "dimension gap" 
                }
                "configuration": dict{
                    "multi objectives": dict{
                        "objs1"
                        "objs2"
                        "objs3"
                    }
                    "test case"
                    "number of test case"
                }
        }
    }
"""


def save_test_results(check_multi_objs, *args):
    objectiveValueBestIndAllGenAllParas = args[0]
    algo_name = sys.argv[2]
    seed = int(sys.argv[3])
    dataSetType = sys.argv[4]
    dataSetName = sys.argv[5]
    load_dict = json.load(open("./gp_paras.json", 'r'))
    test_paras = load_dict["test_paras"]
    algo_paras = test_paras[dataSetType][dataSetName][0]
    save_path = "./save/Test/{0}/{1}_{2}".format(algo_name, dataSetType, dataSetName)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = '{0}/seed{1}.json'.format(save_path, seed)
    train_file = "./save/Train/{0}/Jobs20Mas10/seed{1}.json".format(algo_name, seed)
    train_paras = json.load(open(train_file, 'r'))
    configuration = train_paras["configuration"]

    bestIndAllGenObjsAllParas: dict = {}
    for algo_para_count, algo_para in enumerate(algo_paras):
        bestIndAllGenObjs: list = [
            {
                "gen": obj_count+1,
                "objective value": objective_values.item()
            }
            for obj_count, objective_values in enumerate(objectiveValueBestIndAllGenAllParas[:, algo_para_count])
        ]
        algo_configuration: list = [
            {
                "configuration": {
                    "test case": ".".join([dataSetType, dataSetName]),
                    "number of test case": test_paras[dataSetType][dataSetName][1]
                }
            }
        ]
        bestIndAllGenObjsAllParas["training size: Jobs{0}Mas{1}".format(algo_para[0], algo_para[1])] \
            = bestIndAllGenObjs + algo_configuration

    results: dict = {
        "configuration": configuration,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "effectiveness": bestIndAllGenObjsAllParas
    }

    if check_multi_objs:
        # for multi objs
        pfAllGenObjsAllParas: dict = {}
        makespan = args[1]
        structure_complexity = args[2]
        dimension_gap = args[3]
        for algo_para_count, algo_para in enumerate(algo_paras):
            pfAllGenObjs: list = [
                # {
                #     "gen": gen,
                #     "makespan": [float(i) for i in makespan[algo_para_count][gen]],
                #     "structure complexity": [float(i) for i in structure_complexity[algo_para_count][gen]],
                #     "dimension gap": [float(i) for i in dimension_gap[algo_para_count][gen]]
                # }
                # for gen in range(load_dict[algo_name + "_paras"]["num_gen"])

                # adapt for the last genration
                {
                    "gen": load_dict[algo_name + "_paras"]["num_gen"]-1,
                    "makespan": [float(i) for i in makespan[algo_para_count][0]],
                    "structure complexity": [float(i) for i in structure_complexity[algo_para_count][0]],
                    "dimension gap": [float(i) for i in dimension_gap[algo_para_count][0]]
                }
            ]
            algo_configuration: list = [
                {
                    "configuration": {
                        "multi objectives": {
                            "objs1": "makespan",
                            "objs2": "structure complexity",
                            "objs3": "dimension gap"
                        },
                        "test case": ".".join([dataSetType, dataSetName]),
                        "number of test case": test_paras[dataSetType][dataSetName][1]
                    }
                }
            ]
            pfAllGenObjsAllParas["training size: Jobs{0}Mas{1}".format(algo_para[0], algo_para[1])] \
                = pfAllGenObjs + algo_configuration
        results["multi objectives"] = pfAllGenObjsAllParas
    json.dump(results, fp=open(file_path, mode='w'))

# def saveTrainResults(best_ind_all_gen, best_fit_all_gen, train_time, multiple_objective=False):
#     algo_name = sys.argv[2]
#     seed = int(sys.argv[3])
#     num_jobs = int(sys.argv[4])
#     num_mas = int(sys.argv[5])
#     save_path = './save/{0}/Train_jobs{1}_mas{2}'.format(algo_name, num_jobs, num_mas)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     writer = pd.ExcelWriter('{0}/seed{1}.xlsx'.format(save_path, seed))
#     file_name = [i + 1 for i in range(len(best_ind_all_gen))]
#     data_file = pd.DataFrame(file_name, columns=["generation"])
#     data_file.to_excel(writer, sheet_name='Sheet1', index=False)
#     data = pd.DataFrame([list(i) for i in zip(best_ind_all_gen)], columns=["bestInd"])
#     data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1)
#     if multiple_objective:
#         data = pd.DataFrame([i for i in zip(best_fit_all_gen)], columns=["bestFit"])
#     else:
#         data = pd.DataFrame(np.array(best_fit_all_gen), columns=["bestFit"])
#     data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=2)
#     data = pd.DataFrame(sum(train_time), columns=["total_train_time"])
#     data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=3)
#     writer.close()

#
#
# def saveTestResults(check_multi_objs, *args):
#     objectiveValueBestIndAllGenAllParas = args[0]
#     algo_name = sys.argv[2]
#     seed = int(sys.argv[3])
#     dataSetType = sys.argv[4]
#     dataSetName = sys.argv[5]
#     with open("./gp_paras.json", 'r') as load_f:
#         load_dict = json.load(load_f)
#     test_paras = load_dict["test_paras"]
#     algo_paras = test_paras[dataSetType][dataSetName][0]
#     save_path = './TestResults/{0}/{1}_{2}'.format(algo_name, dataSetType, dataSetName)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     writer = pd.ExcelWriter('{0}/seed{1}.xlsx'.format(save_path, seed))
#     gen_count_name = ["generation " + str(i + 1) for i in range(load_dict[algo_name + "_paras"]["num_gen"])]
#     data_file = pd.DataFrame(gen_count_name, columns=["generation"])
#     data_file.to_excel(writer, sheet_name='Sheet1', index=False)
#     for algo_para_count, algo_para in enumerate(algo_paras):
#         data = pd.DataFrame(objectiveValueBestIndAllGenAllParas[:, algo_para_count],
#                                  columns=['TrainingSize: J{0}M{1}'.format(algo_para[0], algo_para[1])])
#         data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=algo_para_count + 1)
#     if check_multi_objs:
#         # for multi objs
#         data_file.to_excel(writer, sheet_name='MultiObjs', index=False)
#         num_gen = len(args[1][0])
#         for algo_para_count, algo_para in enumerate(algo_paras):
#             objValuesAllGen = []
#             for gen in range(num_gen):
#                 objValuesAllGen.append(tuple(zip(args[1][algo_para_count][gen], args[2][algo_para_count][gen], args[3][algo_para_count][gen])))
#             data = pd.DataFrame([i for i in zip(objValuesAllGen)], columns=['TrainingSize: J{0}M{1}'.format(algo_para[0], algo_para[1])])
#             data.to_excel(writer, sheet_name='MultiObjs', index=False, startcol=algo_para_count + 1)
#     writer.close()
