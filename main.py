import sys
import junwei.algorithms.standardGP.main as standardGP
import junwei.GPTest.main as gpTest
import junwei.algorithms.NSGP2.main as NSGP2
"""
    created by JPang 
    date: 5/9/2023
    the main function to run algorithms
    1. for train 
        for example: input parameters ----> [Train standardGP 0 20 10]
    2. for test
        for example: input parameters ----> [Test standardGP 0 PublicBenchmarks e_la]  
"""

if __name__ == '__main__':
    algo_type = sys.argv[1]
    if algo_type == "Train":
        algo = sys.argv[2]
        seed = int(sys.argv[3])
        num_jobs = int(sys.argv[4])
        num_mas = int(sys.argv[5])
        if algo == 'standardGP':
            print('------------standardGP------------')
            standardGP.main(seed)
        elif algo == 'NSGP2':
            print('------------NSGP2-----------')
            NSGP2.main(seed)
        else:
            print("Unknown algorithm:", algo)
    elif algo_type == "Test":
        seed = int(sys.argv[3])
        dataSetType = sys.argv[4]
        dataSetName = sys.argv[5]
        print('------------gpTest-----------\n')
        print('Dataset: ' + dataSetName + '\nAlgorithm name: ' + sys.argv[2])
        gpTest.main(seed, dataSetType, dataSetName)
