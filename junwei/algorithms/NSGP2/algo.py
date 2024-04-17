import copy
import random
import time
from junwei.utils.selection import standardVarOr, NSGP2SelTournament
from junwei.utils.tools import set_fitness_MO, set_toolbox_structure_complexity_dimension_gap
from deap import tools
import numpy as np
import json


def NSGP2(population, toolbox, cxpb, mutpb, ngen, env, stats=None, verbose=__debug__):

    # use the history to check how the single terminal is generated
    # history = tools.History()
    # toolbox.decorate("mate", history.decorator)
    # toolbox.decorate("mutate", history.decorator)
    toolbox.unregister("select")
    toolbox.register("select", NSGP2SelTournament)

    num_pop = len(population)
    toolbox = set_toolbox_structure_complexity_dimension_gap(toolbox)
    env.set_training_env(0)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    population = set_fitness_MO(toolbox, population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    best_ind = []
    best_fit = []
    pop_list = []
    training_time = np.zeros([ngen, 1])
    # Begin the generational process
    population = tools.selNSGA2(population, num_pop)

    # ###############################################
    # # ind_count_rpt = 0
    # # ind_count_niq = 0
    # # for ind in population:
    # #     if str(ind) == 'lazy_primitive_subtract(get_rPT, get_NOR)':
    # #         ind_count_rpt += 1
    # #     if str(ind) == 'lazy_primitive_subtract(get_NIQ, get_NOR)':
    # #         ind_count_niq += 1
    # # print("num lazy_primitive_subtract(get_rPT, get_NOR): " + str(ind_count_rpt))
    # # print("num lazy_primitive_subtract(get_NIQ, get_NOR): " + str(ind_count_niq))
    dict_list: dict = {}
    count_dict: dict = {}
    name_dict: dict = {}
    for ind in population:
        if str(ind) in name_dict:
            name_dict[str(ind)] += 1
        else:
            name_dict[str(ind)] = 1
        if len(ind) in count_dict:
            count_dict[len(ind)] += 1
        else:
            count_dict[len(ind)] = 1
    dict_list["gen: 0 count"] = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    dict_list["gen: 0 name"] = dict(sorted(name_dict.items(), key=lambda item: item[1], reverse=True))
    # for ind in population:
    #     setattr(ind, "offspring", False)
    # ###############################################

    for gen in range(1, ngen + 1):
        env.set_training_env(gen - 1)
        start_time = time.time()

        offspring = toolbox.select(toolbox, population, num_pop)

        offspring = standardVarOr(offspring, toolbox, cxpb, mutpb)
        # for ind_idx in range(0, len(offspring), 2):
        #     if random.random() <= cxpb:
        #         offspring[ind_idx], offspring[ind_idx + 1] = toolbox.mate(offspring[ind_idx], offspring[ind_idx + 1])
        #
        #         # ############################################
        #         # # try the trick
        #         # while offspring[ind_idx] in population or offspring[ind_idx+1] in population:
        #         #     offspring[ind_idx], offspring[ind_idx + 1] = toolbox.mate(offspring[ind_idx],
        #         #                                                               offspring[ind_idx + 1])
        #         ############################################
        #
        #     if random.random() <= mutpb:
        #         offspring[ind_idx], = toolbox.mutate(offspring[ind_idx])
        #     if random.random() <= mutpb:
        #         offspring[ind_idx + 1], = toolbox.mutate(offspring[ind_idx + 1])

        ###############################################
        # ind_count_rpt = 0
        # for ind in offspring:
        #     if str(ind) == 'lazy_primitive_subtract(get_rPT, get_NOR)':
        #         ind_count_rpt += 1
        # print("num lazy_primitive_subtract(get_rPT, get_NOR): " + str(ind_count_rpt))
        ###############################################

        ###############################################
        ###############################################

        #################################################################################
        # non_duplicate_list_population = []
        # for ind in population:
        #     if str(ind) not in non_duplicate_list_population:
        #         non_duplicate_list_population.append(str(ind))
        # print("num of non-duplicate inds in the population: " + str(len(non_duplicate_list_population)))
        #
        # non_duplicate_list_offspring = []
        # for ind in offspring:
        #     if str(ind) not in non_duplicate_list_population:
        #         non_duplicate_list_offspring.append(str(ind))
        # print("num of non-duplicate inds in the offspring: " + str(len(non_duplicate_list_offspring)))
        #################################################################################

        population = population + offspring
        nevals = len(population)
        population = set_fitness_MO(toolbox, population)
        population = tools.selNSGA2(population, num_pop)

        ###############################################

        # ind_count = 0
        # for ind in population:
        #     if ind.offspring:
        #         ind_count += 1
        # print("num offspring reserved: " + str(ind_count))
        #
        # for ind in offspring:
        #     setattr(ind, "offspring", False)
        count_dict: dict = {}
        name_dict: dict = {}
        for ind in population:
            if str(ind) in name_dict:
                name_dict[str(ind)] += 1
            else:
                name_dict[str(ind)] = 1
            if len(ind) in count_dict:
                count_dict[len(ind)] += 1
            else:
                count_dict[len(ind)] = 1
        dict_list[f"gen: {gen} count"] = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        dict_list[f"gen: {gen} name"] = dict(sorted(name_dict.items(), key=lambda item: item[1], reverse=True))
        # count_dict = {}
        # for ind in population:
        #     # if str(ind) in count_dict:
        #     #     count_dict[str(ind)] += 1
        #     # else:
        #     #     count_dict[str(ind)] = 1
        #     if len(ind) in count_dict:
        #         count_dict[len(ind)] += 1
        #     else:
        #         count_dict[len(ind)] = 1
        # sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        # dict_list["gen: {}".format(gen)] = sorted_dict
        ###############################################

        training_time[gen - 1] = time.time() - start_time
        first_pareto_front = tools.sortNondominated(population, num_pop)[0]

        ###############################################
        # non_duplicate_list = []
        # ind_objs = []
        # for ind in population:
        #     if str(ind) not in non_duplicate_list:
        #         non_duplicate_list.append(str(ind))
        #         ind_objs.append(ind.original_fitness)
        # # print("before print gen information: " + str([str(i) for i in non_duplicate_list]))
        # print("num inds: " + str(len(non_duplicate_list)))
        # print("non-normalised objs: " + str([objs[0] for objs in ind_objs]))
        ###############################################

        # fit_list  = []
        # first_pareto_front_non_duplicate = []
        # first_pareto_front_non_duplicate_fitness = []
        # for ind in first_pareto_front:
        #     if str(ind) not in first_pareto_front_non_duplicate:
        #         first_pareto_front_non_duplicate.append(str(ind))
        #         first_pareto_front_non_duplicate_fitness.append(ind.original_fitness)
        best_ind.append([str(ind) for ind in first_pareto_front])
        best_fit.append([ind.original_fitness for ind in first_pareto_front])
        pop_list.append([str(ind) for ind in population])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)
        if verbose:
            print(logbook.stream)
        print(training_time[gen - 1])

    return pop_list, best_ind, best_fit, training_time

