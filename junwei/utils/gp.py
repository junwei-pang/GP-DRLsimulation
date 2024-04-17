import time
from functools import partial
from collections import defaultdict
from operator import eq, lt
import random

import deap
from deap import gp
import json
import sys

"""
    created by JPang
    Date: 15/11/2023
    considering that there are lots of differences between DEAP and ECJ, 
    this file would over-ride some functions in deap.gp.py 
    
    
    created by JPang
    Date: 14/11/2023
    consider that there are some differences between ECJ and DEAP. So, the node selection needs to be re-wrote  
    1. the prob for selecting the function and terminals is set to 0.9 and 0.1 respectively 
    2. the mutation is changed to gp.genGrow and min: 1 max: 4
    3. change all the paras related to depth from original value to original value - 1
"""


def cx_biased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    # Determine whether to keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        # Set does not support indexing
        type_ = random.sample(common_types, 1)[0]
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def node_selector(individual: gp.PrimitiveTree) -> int:
    terminal_idx_list: list[int] = []
    nonterminal_idx_list: list[int] = []
    rnd: float = random.random()
    for node_idx, node in enumerate(individual):
        if isinstance(node, deap.gp.Terminal):
            terminal_idx_list.append(node_idx)
        elif isinstance(node, deap.gp.Primitive):
            nonterminal_idx_list.append(node_idx)
        else:
            raise ValueError("This node is not belong to terminal or function")
    # according to the ECJ, the prob to select terminal is set ot 0.1 and the prob to select non-terminal is set to 0.9
    prob_select_terminal: float = json.load(open("./gp_paras.json", "r"))["gp_paras"]["prob_select_terminal"]
    if rnd < prob_select_terminal:  # pick the terminal node
        return random.choice(terminal_idx_list)
    elif nonterminal_idx_list:
        return random.choice(nonterminal_idx_list)
    else:
        return 0

def mut_biased_ECJ(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = node_selector(individual)
    # index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def generate(pset, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < 1/(1+pset.prims_count))

    return generate(pset, min_, max_, condition, type_)


def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def isclass(object):
    """Return true if the object is a class.

    Class objects provide these attributes:
        __doc__         documentation string
        __module__      name of module in which this class was defined"""
    return isinstance(object, type)
