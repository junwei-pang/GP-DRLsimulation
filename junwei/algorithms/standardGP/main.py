import random
from junwei.env.env_manager import StaticFlexibleEnvManager
import numpy as np
from deap import tools
from junwei.algorithms.standardGP.algo import standardGP
from junwei.utils import save_file
from junwei.utils.init import standardGPInitializer, getStandardGPParas

"""
    Created by JPang
    Train the GP for the static flexible job shop problems with the same simulation.

    NOTE: 
    1. GP is designed to select the best O-M pair from the global sequence at every stage 
"""


def initSimpleGPAndEnv():
    stat_flex_env_manager = StaticFlexibleEnvManager()
    pset, toolbox = standardGPInitializer(stat_flex_env_manager)
    return pset, toolbox, stat_flex_env_manager


def main(seed):
    pset, toolbox, stat_flex_env_manager = initSimpleGPAndEnv()
    np.random.seed(seed)
    random.seed(seed)
    # load the data
    num_elitism, num_gen, num_pop, cxpb, mutpb = getStandardGPParas()

    pop = toolbox.population(n=num_pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop_list, bestIndAllGen, bestFitAllGen, training_time = standardGP(pop, toolbox, cxpb, mutpb, num_elitism, num_gen,
                                                                       env=stat_flex_env_manager, stats=stats,
                                                                       halloffame=hof)
    save_file.save_train_results(pset, toolbox, pop_list, bestIndAllGen, bestFitAllGen, training_time)
    # pool.close()
