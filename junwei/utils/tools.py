import copy
import json
import random
import numpy as np

"""
    created by JPang 
    list all the useful functions, like a warehouse 
"""


def get_ind_complexity(ind, structure_complexity_dict):
    indStructureComplexityValue = 0
    for node in ind:
        if node.name in structure_complexity_dict:
            indStructureComplexityValue += structure_complexity_dict[node.name]
    return indStructureComplexityValue


def get_ind_dimension_gap(ind, dimension_dict):
    dimensionGapVector = []
    dimensionVector = []
    ind_copied = copy.deepcopy(ind)
    while ind_copied:
        node = ind_copied.pop()
        if node.name in dimension_dict:
            dimensionVector.append(dimension_dict[node.name])
        else:
            child1 = dimensionVector.pop()
            child2 = dimensionVector.pop()
            if node.name == "lazy_protected_div":
                vector = [a - b for a, b in zip(child1, child2)]
            elif node.name == "lazy_primitive_multiply":
                vector = [a + b for a, b in zip(child1, child2)]
            else:
                vector = [(a + b) / 2 for a, b in zip(child1, child2)]
                dimensionGapVector.append(sum([abs(a - b) for a, b in zip(child1, child2)]))
            dimensionVector.append(vector)
    return sum(dimensionGapVector)


def set_fitness_MO(toolbox, individuals):
    performance = list(toolbox.map(toolbox.evaluate, individuals))
    min_performance = min(performance)[0]
    performance_gap = max(performance)[0] - min_performance
    structure_complexity = list(toolbox.map(toolbox.get_structure_complexity, individuals))
    min_structure_complexity = min(structure_complexity)
    structure_complexity_gap = max(structure_complexity) - min_structure_complexity
    dimension_gap = list(toolbox.map(toolbox.get_dimension_gap, individuals))
    min_dimension_gap = min(dimension_gap)
    dimension_gap_gap = max(dimension_gap) - min_dimension_gap

    for ind_count, ind in enumerate(individuals):
        ind.fitness.values = (performance[ind_count][0], structure_complexity[ind_count], dimension_gap[ind_count])
        setattr(ind, "original_fitness", ind.fitness.values)
        ind.fitness.values = (
            (performance[ind_count][0] - min_performance) / performance_gap,
            (structure_complexity[ind_count] - min_structure_complexity) / structure_complexity_gap,
            (dimension_gap[ind_count] - min_dimension_gap) / dimension_gap_gap
        )
    return individuals


def set_toolbox_structure_complexity_dimension_gap(toolbox):
    structure_complexity_dict = json.load(open("./gp_paras.json", "r"))["structure_complexity_dict"]["values"]
    dimension_dict = json.load(open("./gp_paras.json", "r"))["dimension_dict"]["values"]
    toolbox.register("get_structure_complexity", get_ind_complexity,
                     structure_complexity_dict=structure_complexity_dict)
    toolbox.register("get_dimension_gap", get_ind_dimension_gap, dimension_dict=dimension_dict)
    return toolbox


def reset_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
