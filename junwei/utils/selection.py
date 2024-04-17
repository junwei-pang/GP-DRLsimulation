import copy
import numpy as np
import random
from deap import gp, tools

"""
    created by JPang
    date: 12/9/2023
    all the created selection method 
"""


# standard select parents to do crossover, mutation and reservation
def standardVarOr(population, toolbox, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = [toolbox.clone(ind) for ind in population]
    end = len(offspring) - 1

    cur = 0
    while cur <= end:
        op_choice = random.random()
        if op_choice < cxpb:  # Apply crossover
            if cur <= end - 1:
                offspring[cur], offspring[cur + 1] = toolbox.mate(offspring[cur], offspring[cur + 1])
                del offspring[cur].fitness.values, offspring[cur + 1].fitness.values
                cur += 1
            else:
                del offspring[cur].fitness.values
        elif op_choice < cxpb + mutpb:  # Apply mutation
            offspring[cur], = toolbox.mutate(offspring[cur])
            del offspring[cur].fitness.values
        else:  # Apply reproduction
            del offspring[cur].fitness.values

        cur += 1

    return offspring


def standardGPSelectElitism(toolbox, num_elitism, population):
    pop = [toolbox.clone(ind) for ind in population]
    fitness = [pop[i].fitness.values for i in range(len(pop))]
    fitness = np.array(fitness).T
    idx_elitism = np.argpartition(fitness[0], num_elitism)[:num_elitism]
    elitism = [population[i] for i in idx_elitism]
    return elitism


def NSGP2TournamentFor2Individuals(ind1, ind2):
    if ind1.fitness.dominates(ind2.fitness):
        return ind1
    elif ind2.fitness.dominates(ind1.fitness):
        return ind2

    if ind1.fitness.crowding_dist < ind2.fitness.crowding_dist:
        return ind2
    elif ind1.fitness.crowding_dist > ind2.fitness.crowding_dist:
        return ind1

    if random.random() <= 0.5:
        return ind1
    return ind2


def NSGP2SelTournament(toolbox, individuals, k):
    """Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4 only if k is equal to the length of individuals.
    Starting from the beginning of the selected individuals, two consecutive
    individuals will be different (assuming all individuals in the input list
    are unique). Each individual from the input list won't be selected more
    than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select. Must be less than or equal
              to len(individuals).
    :returns: A list of selected individuals.
    """

    if k > len(individuals):
        raise ValueError("selTournamentDCD: k must be less than or equal to individuals length")

    if k == len(individuals) and k % 4 != 0:
        raise ValueError("selTournamentDCD: k must be divisible by four if k == len(individuals)")

    individuals_1_copied: list[gp.PrimitiveTree] = []
    individuals_2_copied: list[gp.PrimitiveTree] = []
    for ind_idx, ind in enumerate(individuals):
        individuals_1_copied.append(toolbox.clone(ind))
        individuals_2_copied.append(toolbox.clone(ind))
        individuals_1_copied[ind_idx].fitness.crowding_dist = individuals[ind_idx].fitness.crowding_dist
        individuals_2_copied[ind_idx].fitness.crowding_dist = individuals[ind_idx].fitness.crowding_dist
        # setattr(individuals_1_copied[ind_idx], "fitness.crowding_distance", )
        # setattr(individuals_2_copied[ind_idx], "fitness.crowding_distance", individuals[ind_idx].fitness.crowding_distance)
    # individuals_1 = random.sample(individuals, len(individuals))
    # individuals_2 = random.sample(individuals, len(individuals))
    individuals_1 = [random.choice(individuals_1_copied) for _ in range(len(individuals))]
    individuals_2 = [random.choice(individuals_2_copied) for _ in range(len(individuals))]

    chosen = []
    for i in range(0, k, 4):
        chosen.append(NSGP2TournamentFor2Individuals(individuals_1[i], individuals_1[i + 1]))
        chosen.append(NSGP2TournamentFor2Individuals(individuals_1[i + 2], individuals_1[i + 3]))
        chosen.append(NSGP2TournamentFor2Individuals(individuals_2[i], individuals_2[i + 1]))
        chosen.append(NSGP2TournamentFor2Individuals(individuals_2[i + 2], individuals_2[i + 3]))

    return chosen


def tournament_2Individuals_archive_population(individual1, individual2, first_pareto_front):
    if hasattr(individual1, "from_archive") and hasattr(individual2, "from_archive"):
        return random.choice([individual1, individual2])
    elif hasattr(individual1, "from_archive") and not hasattr(individual2, "from_archive"):
        if individual2 in first_pareto_front:
            return individual2
        else:
            return individual1
    elif hasattr(individual2, "from_archive") and not hasattr(individual1, "from_archive"):
        if individual1 in first_pareto_front:
            return individual1
        else:
            return individual2
    else:
        return NSGP2TournamentFor2Individuals(individual1, individual2)

def selection_archive_population(toolbox, population, archive, k):
    first_pareto_front: list = tools.sortNondominated(population, len(population))[0]
    # if k > len(individuals):
    #     raise ValueError("selTournamentDCD: k must be less than or equal to individuals length")
    #
    # if k == len(individuals) and k % 4 != 0:
    #     raise ValueError("selTournamentDCD: k must be divisible by four if k == len(individuals)")

    individuals_1_copied: list[gp.PrimitiveTree] = []
    individuals_2_copied: list[gp.PrimitiveTree] = []
    for ind_idx, ind in enumerate(population):
        individuals_1_copied.append(toolbox.clone(ind))
        individuals_2_copied.append(toolbox.clone(ind))
        individuals_1_copied[ind_idx].fitness.crowding_dist = population[ind_idx].fitness.crowding_dist
        individuals_2_copied[ind_idx].fitness.crowding_dist = population[ind_idx].fitness.crowding_dist
        # setattr(individuals_1_copied[ind_idx], "fitness.crowding_distance", )
        # setattr(individuals_2_copied[ind_idx], "fitness.crowding_distance", individuals[ind_idx].fitness.crowding_distance)
    # individuals_1 = random.sample(individuals, len(individuals))
    # individuals_2 = random.sample(individuals, len(individuals))

    archive_1_copied = [toolbox.clone(ind) for ind in archive]
    for ind in archive_1_copied:
        setattr(ind, "from_archive", True)

    archive_2_copied = [toolbox.clone(ind) for ind in archive]
    for ind in archive_2_copied:
        setattr(ind, "from_archive", True)

    individuals_1_copied = individuals_1_copied + archive_1_copied
    individuals_2_copied = individuals_2_copied + archive_2_copied

    individuals_1 = [random.choice(individuals_1_copied) for _ in range(len(individuals_1_copied))]
    individuals_2 = [random.choice(individuals_2_copied) for _ in range(len(individuals_2_copied))]

    chosen = []
    for i in range(0, k, 4):
        chosen.append(tournament_2Individuals_archive_population(individuals_1[i], individuals_1[i + 1], first_pareto_front))
        chosen.append(tournament_2Individuals_archive_population(individuals_1[i + 2], individuals_1[i + 3], first_pareto_front))
        chosen.append(tournament_2Individuals_archive_population(individuals_2[i], individuals_2[i + 1], first_pareto_front))
        chosen.append(tournament_2Individuals_archive_population(individuals_2[i + 2], individuals_2[i + 3], first_pareto_front))

    for ind in chosen:
        if hasattr(ind, "from_archive"):
            delattr(ind, "from_archive")

    return chosen
