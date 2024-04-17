import time
from deap import tools
import numpy as np
from junwei.utils.selection import standardGPSelectElitism, standardVarOr


# delet the fitness values of individuals
def invalid_and_evaluation(toolbox, pop):
    for i in range(len(pop)):
        del pop[i].fitness.values
    fitnesses = toolbox.multiprocessing(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return pop


def standardGP(population, toolbox, cxpb, mutpb, num_elitism, ngen, env, stats=None, halloffame=None,
               verbose=__debug__):
    env.set_training_env(0)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population]
    fitnesses = toolbox.multiprocessing(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best_ind: list = []
    best_fit: list = []
    pop_list: list = []
    training_time = np.zeros([ngen, 1])
    # Begin the generational process

    for gen in range(1, ngen+1):
        env.set_training_env(gen-1)
        start_time = time.time()
        elitism = standardGPSelectElitism(toolbox, num_elitism, population)

        elitism = invalid_and_evaluation(toolbox, elitism)

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - num_elitism)

        # Vary the pool of individuals
        offspring = standardVarOr(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        offspring = invalid_and_evaluation(toolbox, offspring)

        population[:] = elitism + offspring

        training_time[gen-1] = time.time() - start_time
        # pre_best = copy.deepcopy(halloffame[0])
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.clear()
            halloffame.update(population)

        # print("before gen the len of best ind: " + str(len(halloffame[0])))

        best_ind.append(str(halloffame[0]))
        best_fit.append(list(halloffame.keys[0].values)[0])
        pop_list.append([str(ind) for ind in population])

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        print(training_time[gen-1])

    return pop_list, best_ind, best_fit, training_time
