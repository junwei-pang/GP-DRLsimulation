import json
import sys
import numpy as np
import random
from deap import gp
from junwei.utils.tools import reset_seed
from junwei.env.case_generator import CaseGenerator
from junwei.env.env_generator import StaticFlexibleEnv4PC

"""
    created by JPang
    Date: 03/10/2023
    set the archive to store useful individuals
"""


class Archive:
    def __init__(self):
        with open("./gp_paras.json", 'r') as load_f:
            load_dict = json.load(load_f)
        self.archive_ind_list: list[gp.PrimitiveTree] = []
        self.algo_seed: int = int(sys.argv[3])
        self.pc_instance_seed: int = 0
        self.pc_instance_rotation_seed: int = 0
        # self.shuffle_seed = 8295342
        # self.num_decision_situation: int = 10  # in dynamic situation, the number is 20
        # self.min_queue_length: int = 4  # in dynamic situation, the number is 8

        self.num_jobs: int = int(sys.argv[4])
        self.num_mas: int = int(sys.argv[5])
        self.opes_per_job_min: int = int(self.num_mas * 0.8)
        self.opes_per_job_max: int = int(self.num_mas * 1.2)

        self.situations_for_PC: dict[str, list] = {}

    def set_archive_env_for_pc(self, env, pset, num_archive_env=1):
        for archive_count in range(num_archive_env):
            reset_seed(self.pc_instance_seed + archive_count * self.pc_instance_rotation_seed)
            case = CaseGenerator(
                self.num_jobs, self.num_mas, self.opes_per_job_min, self.opes_per_job_max, flag_same_opes=False
            )
            env.archive_pc_envs.append(StaticFlexibleEnv4PC(case=case))
        env.set_archive_env(0)
        env.cur_env.get_situation_for_calculate_pc(pset)
        reset_seed(self.algo_seed)

    def archive_for_pc_update_smaller_size(self, toolbox, env, pset, non_dominated_inds, training_env_count=0):
        env.set_archive_env(0)
        non_dominated_inds_copied = [toolbox.clone(ind) for ind in non_dominated_inds]
        for ind in non_dominated_inds_copied:
            if not self.archive_ind_list:
                archive_str = [str(ind) for ind in self.archive_ind_list]
                if str(ind) in archive_str:
                    break
            duplicate: bool = False
            ind_pc = env.cur_env.get_pc(ind, pset)
            setattr(ind, "phenotypic_characterisation", ind_pc)
            if not self.archive_ind_list:
                setattr(ind, "from_archive", True)
                self.archive_ind_list.append(ind)
            else:
                for archive_ind_count, archive_ind in enumerate(self.archive_ind_list):
                    if sum([0 if a == b else 1 for a, b in zip(ind_pc, archive_ind.phenotypic_characterisation)]) == 0 and len(ind) < len(archive_ind):
                        duplicate = True
                        setattr(ind, "from_archive", True)
                        self.archive_ind_list[archive_ind_count] = ind
                        break
            if not duplicate:
                setattr(ind, "from_archive", True)
                self.archive_ind_list.append(ind)

        env.set_training_env(training_env_count)
