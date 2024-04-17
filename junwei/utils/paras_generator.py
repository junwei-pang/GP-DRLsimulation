"""
    created by JPang
    Date: 12/9/2023
    generate the text needed to do parallel computing
"""
# savePath = "../seeds_30.txt"
# train_case_pool = [[10, 5], [15, 10], [20, 5], [20, 10]]
# test_case_pool = {"SyntheticInstancesWithSameSize": ["1005", "1510", "2005", "2010"],
#                   "SyntheticInstancesWithLargeSize": ["3010", "4010", "5020"],
#                   "PublicBenchmarks": ["e_la", "Mk", "r_la", "v_la"]}
# num_seeds = 30

"""
    for train 
"""
# algo_name = "alpha_dominanceGP"
#
# with open(savePath, "w") as file:
#     for j in range(len(train_case_pool)):
#         train_paras = train_case_pool[j]
#         for seed in range(num_seeds):
#             line = f"Train {algo_name} {seed} {train_paras[0]} {train_paras[1]}\n"
#             file.write(line)


"""
    for test
"""
# algo_name = "alpha_dominanceGP"
# with open(savePath, "w") as file:
#     for testCaseType in test_case_pool:
#         test_case = test_case_pool[testCaseType]
#         for case in test_case:
#             for seed in range(num_seeds):
#                 line = f"Test {algo_name} {seed} {testCaseType} {case}\n"
#                 file.write(line)


savePath = "../cm.txt"
fronts = 27
algo_name = "alpha_dominanceGP"
with open(savePath, "w") as file:
    for parent_front in range(fronts):
        line = f"mutation {parent_front} {parent_front} 20 10\n"
        file.write(line)
        for donor_front in range(fronts):
            line = f"crossover {parent_front} {donor_front} 20 10\n"
            file.write(line)


