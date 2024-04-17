import pygraphviz as pgv
import pandas as pd
import re
import json

"""
    created by JPang
    date: 17/08/2023
    visualisation for experimental results
    1. plot the tree by pygraphviz
"""


def is_function(node):
    valid_functions = {'lazy_primitive_add', 'lazy_primitive_subtract', 'lazy_primitive_multiply', 'lazy_protected_div',
                       'lazy_primitive_maximum', 'lazy_primitive_minimum'}
    return node in valid_functions


def trans2function(node):
    function_mapping = {
        'lazy_primitive_add': '+',
        'lazy_primitive_subtract': '-',
        'lazy_primitive_multiply': '*',
        'lazy_protected_div': '/',
        'lazy_primitive_maximum': 'max',
        'lazy_primitive_minimum': 'min'
    }
    return function_mapping.get(node, False)


def graph(expr):
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = trans2function(node) if is_function(node) else node.replace('get_', '')
        if is_function(node):
            stack.append([i, 2])
        else:
            stack.append([i, 0])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


if __name__ == '__main__':
    # path = 'C:/Users/pangj/Desktop/Results/Train/standardGP_old/Train_jobs20_mas10/seed0.xlsx'
    # data = pd.read_excel(path, sheet_name="Sheet1")
    # ind = data.bestInd.iloc[-1]

    json_file = json.load(open("C:/Users/pangj/Desktop/Results/Train/alpha_dominanceGP/Jobs20Mas10/seed22.json", "r"))
    ind_list = json_file["population"][49]["pareto fronts"]["inds"]
    ind = ind_list[10]
    # ind = "lazy_primitive_add(lazy_primitive_multiply(get_PT, get_NOR), lazy_primitive_maximum(get_NPT, get_WKR))"
    ind = [i for i in re.split(r"[(),\s]", ind) if i]
    nodes, edges, labels = graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes, shape="box", fontsize=16, width=0.5, height=0.3)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("C:/Users/pangj/Desktop/tree.pdf")




