import deap.gp
from deap import gp
from deap import tools
from deap import base
from deap import creator
import numpy as np
from functools import partial
import operator
import copy
import random
import json
import pandas as pd
import sys
import multiprocessing
from junwei.utils.gp import cx_biased, genHalfAndHalf, genGrow, mut_biased_ECJ

"""
    created by JPang
    Date: 27/06/2023
    the basic component of tree-based GP, i.ie. functions
"""


def getMOGPParas(algo_name):
    with open("./gp_paras.json", 'r') as load_f:
        load_dict = json.load(load_f)
    alphaDominanceGPParas = load_dict[algo_name]
    num_gen = alphaDominanceGPParas["num_gen"]
    num_pop = alphaDominanceGPParas["num_pop"]
    cxpb = alphaDominanceGPParas["cxpb"]
    mutpb = alphaDominanceGPParas["mutpb"]
    return num_gen, num_pop, cxpb, mutpb


def getStandardGPParas():
    with open("./gp_paras.json", 'r') as load_f:
        load_dict = json.load(load_f)
    standardGP_paras = load_dict["standardGP_paras"]
    num_elitism = standardGP_paras["num_elitism"]
    num_gen = standardGP_paras["num_gen"]
    num_pop = standardGP_paras["num_pop"]
    cxpb = standardGP_paras["cxpb"]
    mutpb = standardGP_paras["mutpb"]
    return num_elitism, num_gen, num_pop, cxpb, mutpb


def primitive_add(out1, out2):
    return np.add(out1(), out2())


def lazy_primitive_add(out1, out2):
    return partial(primitive_add, out1, out2)


def primitive_subtract(out1, out2):
    return np.subtract(out1(), out2())


def lazy_primitive_subtract(out1, out2):
    return partial(primitive_subtract, out1, out2)


def primitive_multiply(out1, out2):
    return np.multiply(out1(), out2())


def lazy_primitive_multiply(out1, out2):
    return partial(primitive_multiply, out1, out2)


def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left(), right())
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


def lazy_protected_div(out1, out2):
    return partial(protected_div, out1, out2)


def primitive_maximum(out1, out2):
    return np.maximum(out1(), out2())


def lazy_primitive_maximum(out1, out2):
    return partial(primitive_maximum, out1, out2)


def primitive_minimum(out1, out2):
    return np.minimum(out1(), out2())


def lazy_primitive_minimum(out1, out2):
    return partial(primitive_minimum, out1, out2)


def init_primitives(pset, env_manager):
    pset.addPrimitive(lazy_primitive_add, 2)
    pset.addPrimitive(lazy_primitive_subtract, 2)
    pset.addPrimitive(lazy_primitive_multiply, 2)
    pset.addPrimitive(lazy_protected_div, 2)
    pset.addPrimitive(lazy_primitive_maximum, 2)
    pset.addPrimitive(lazy_primitive_minimum, 2)
    pset.addTerminal(env_manager.get_PT)
    pset.addTerminal(env_manager.get_OWT)
    pset.addTerminal(env_manager.get_NPT)
    pset.addTerminal(env_manager.get_WKR)
    pset.addTerminal(env_manager.get_NOR)
    pset.addTerminal(env_manager.get_WIQ)
    pset.addTerminal(env_manager.get_NIQ)
    pset.addTerminal(env_manager.get_NOS)
    pset.addTerminal(env_manager.get_MWT)
    pset.addTerminal(env_manager.get_rPT)
    pset.addTerminal(env_manager.get_rWIQ)
    pset.addTerminal(env_manager.get_rNIQ)

    # delete rMWT
    # pset.addTerminal(env_manager.get_rMWT)
    return pset

"""
    created by JPang
    Data: 27/06/2023
    initialise the standardGP
    
    Date: 15/11/2023
    some changes needs to be done here 
    2. the mutation is changed to gp.genGrow and min: 1 max: 4
    3. change all the paras related to depth from original value to original value - 1
"""


def standardGPInitializer(env_manager, num_objs=None):
    pset = gp.PrimitiveSet("Main", 0)
    pset = init_primitives(pset=pset, env_manager=env_manager)
    if num_objs is None:
        weights = (-1.0,)
    else:
        weights = tuple([-1] * num_objs)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", env_manager.get_objective_value, pset=pset)

    cores = 1
    print("core:" + str(cores))
    pool = multiprocessing.Pool(cores)
    toolbox.register("multiprocessing", pool.map)

    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", cx_biased, termpb=json.load(open("./gp_paras.json", "r"))["gp_paras"]["prob_select_terminal"])
    toolbox.register("expr_mut", genGrow, min_=0, max_=3)
    # toolbox.register("expr_mut", gp.genGrow, min_=1, max_=4)
    toolbox.register('mutate', mut_biased_ECJ, expr=toolbox.expr_mut, pset=pset)

    # toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("expr", genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # limit the maximum depth
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
    # toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    # toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    return pset, toolbox

def getPoolPathAndInds(seed, algo_para, pset):
    functions = [i for i in pset.primitives[object]]
    terminals = [i for i in pset.terminals[object]]
    pool = functions + terminals
    pool_name = [i.name for i in pool]
    algo_name = sys.argv[2]
    gpIndPath = "./save/{0}/Train_jobs{1}_mas{2}/seed{3}.xlsx".format(algo_name, algo_para[0], algo_para[1], seed)
    df = pd.read_excel(gpIndPath, sheet_name="Sheet1")
    load_inds = df.bestInd
    return pool, pool_name, load_inds

def standardGPFromStr2Ind(seed, algo_para, pset):
    algo_name = sys.argv[2]
    gpIndPath = "./save/Train/{0}/Jobs{1}Mas{2}/seed{3}.json".format(algo_name, algo_para[0], algo_para[1], seed)
    data = json.load(open(gpIndPath, 'r'))
    bestIndsAllGen = [i["best ind in each generation"]['ind'] for i in data["population"]]
    inds: list = []
    for ind in bestIndsAllGen:
        ind_converted = gp.PrimitiveTree.from_string(ind, pset)
        inds.append(ind_converted)
    return inds


def multiObjsGPFromStr2Ind(seed, algo_para, pset):
    algo_name = sys.argv[2]
    gpIndPath = "./save/Train/{0}/Jobs{1}Mas{2}/seed{3}.json".format(algo_name, algo_para[0], algo_para[1], seed)
    data = json.load(open(gpIndPath, 'r'))
    pf_all_gen = data['population']
    pf_fits = [pf_each_gen['pareto fronts']['fitness'] for pf_each_gen in pf_all_gen]
    best_inds = []
    multi_inds = []
    for gen, pf_each_gen in enumerate(pf_all_gen):
        fits_each_gen = pf_fits[gen]
        effectiveness_each_gen = [fit[0] for fit in fits_each_gen]
        best_ind_idx = effectiveness_each_gen.index(min(effectiveness_each_gen))
        pfEachGenFromStr = [gp.PrimitiveTree.from_string(ind, pset) for ind in pf_each_gen['pareto fronts']['inds']]
        # check if there is any duplicate element
        non_duplicate_pf = []
        for ind in pfEachGenFromStr:
            if ind not in non_duplicate_pf:
                non_duplicate_pf.append(ind)
        best_inds.append(pfEachGenFromStr[best_ind_idx])
        multi_inds.append(non_duplicate_pf)
    return best_inds, multi_inds
