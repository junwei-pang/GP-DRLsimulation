import json
import random
import os
from junwei.env.case_generator import CaseGenerator
from junwei.env.env_generator import StaticFlexibleEnv
import numpy as np
import sys


class StaticFlexibleEnvManager:
    def __init__(self):
        self.training_envs: list[StaticFlexibleEnv] = []
        self.test_envs: list[StaticFlexibleEnv] = []
        self.archive_pc_envs: list[StaticFlexibleEnv] = []
        self.cur_env = None
        self.seed_rotation = 10000
        self.seed_instance = 968356

        num_jobs = int(sys.argv[4])
        num_mas = int(sys.argv[5])
        opes_per_job_min = int(num_mas * 0.8)
        opes_per_job_max = int(num_mas * 1.2)
        n_envs = json.load(open("./gp_paras.json", "r"))["env_paras"]["num_envs"]
        # generate n_envs environments for training
        for i in range(n_envs):
            np.random.seed(self.seed_instance)
            random.seed(self.seed_instance)
            # modified by JPang, generate different case for different generation 31/07/2023
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, flag_same_opes=False)
            self.training_envs.append(StaticFlexibleEnv(case=case))
            self.seed_instance += self.seed_rotation

        self.seed_instance = 0
        self.num_decision_situation = 10
        self.minQueuelength = 5
        self.shuffle_seed = 8295342
        np.random.seed(self.seed_instance)
        random.seed(self.seed_instance)

        # case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, flag_same_opes=False)
        # self.training_envs.append(StaticFlexibleEnv(case=case))

    def set_archive_env(self, env_id):
        self.cur_env = self.archive_pc_envs[env_id]

    def set_training_env(self, env_id):
        self.cur_env = self.training_envs[env_id]

    def set_test_env(self, env_id):
        self.cur_env = self.test_envs[env_id]

    def get_PT(self):
        return self.cur_env.get_PT()

    def get_OWT(self):
        return self.cur_env.get_OWT()

    def get_NPT(self):
        return self.cur_env.get_NPT()

    def get_WKR(self):
        return self.cur_env.get_WKR()

    def get_NOR(self):
        return self.cur_env.get_NOR()

    def get_WIQ(self):
        return self.cur_env.get_WIQ()

    def get_NIQ(self):
        return self.cur_env.get_NIQ()

    def get_NOS(self):
        return self.cur_env.get_NOS()

    def get_MWT(self):
        return self.cur_env.get_MWT()

    def get_rPT(self):
        return self.cur_env.get_rPT()

    def get_rWIQ(self):
        return self.cur_env.get_rWIQ()

    def get_rNIQ(self):
        return self.cur_env.get_rNIQ()

    def get_rMWT(self):
        return self.cur_env.get_rMWT()

    def get_objective_value(self, individual, pset):
        return self.cur_env.evlMakeSpan(individual=individual, pset=pset)

    def piecewise_get_objective_value(self, individual, pset, time=None):
        return self.cur_env.pwEvlMakeSpan(individual=individual, pset=pset, time=time)

    def set_draw_gantt(self):
        self.cur_env.draw_gantt = True


class StaticFlexibleEnvManager4Test(StaticFlexibleEnvManager):
    def __init__(self, dataset_name, dataset_size):
        self.test_envs = []
        data_path = "./data_test/{0}/".format(dataset_name)
        num_ins = dataset_size
        test_files = os.listdir(data_path)
        test_files = test_files[:num_ins]
        for count in range(num_ins):
            test_file = data_path + test_files[count]
            self.test_envs.append(StaticFlexibleEnv(case=test_file, data_source='file'))


