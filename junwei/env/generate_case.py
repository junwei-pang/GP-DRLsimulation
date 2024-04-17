from case_generator import CaseGenerator
import os
"""
    created by JPang
    data: 08/08/2023
    generate the test case
"""


def main():
    num_jobs = [40, 50, 100]
    num_mas = [20, 30]
    for i in num_jobs:
        for j in num_mas:
            data_path = '../data_test/{0}{1}'.format(i, j)
            os.mkdir(data_path)
            opes_per_job_min = int(j * 0.8)
            opes_per_job_max = int(j * 1.2)
            case = CaseGenerator(i, j, opes_per_job_min, opes_per_job_max, path=data_path, flag_same_opes=False,
                                 flag_doc=True)
            for k in range(100):
                case.get_case(idx=k)


if __name__ == '__main__':
    main()
