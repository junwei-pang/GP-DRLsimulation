import random
from junwei.env.env_manager import StaticFlexibleEnvManager
import numpy as np
from deap import tools
from junwei.utils import save_file
from junwei.utils.init import standardGPInitializer, getMOGPParas
import json
from junwei.algorithms.NSGP2.algo import NSGP2


"""
    created by JPang 
    Date: 10/10/2023
    nsga2 for multi-objective optimisation with multiple interpretability measure 
    obj1: performance, obj2: complexity, obj3: dimension gap 
"""


def initMOGPAndEnv():
    stat_flex_env_manager = StaticFlexibleEnvManager()
    with open("./gp_paras.json", 'r') as load_f:
        load_dict = json.load(load_f)
    num_objs = load_dict["NSGP2_paras"]["num_objs"]
    pset, toolbox = standardGPInitializer(stat_flex_env_manager, num_objs=num_objs)
    return pset, toolbox, stat_flex_env_manager


def main(seed):
    pset, toolbox, stat_flex_env_manager = initMOGPAndEnv()
    np.random.seed(seed)
    random.seed(seed)
    # load the data
    num_gen, num_pop, cxpb, mutpb = getMOGPParas("NSGP2_paras")
    pop = toolbox.population(n=num_pop)
    stats = tools.Statistics(lambda ind: ind.original_fitness[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop_list, bestIndAllGen, bestFitAllGen, training_time = NSGP2(pop, toolbox, cxpb, mutpb, num_gen,
                                                        env=stat_flex_env_manager, stats=stats)
    save_file.save_train_results(pset, toolbox, pop_list, bestIndAllGen, bestFitAllGen, training_time)
