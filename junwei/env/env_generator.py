from junwei.env.load_data import nums_detec
import numpy as np
import copy
from deap import gp
from collections import defaultdict
import re
from junwei.utils.tools import reset_seed
import sys
import random


class StaticFlexibleEnv:
    def __init__(self, case, data_source='case'):
        self.current_ma: int = 0
        self.current_job: int = 0
        self.current_ope: int = 0
        self.num_opes: int = 0
        self.eligible_opes = []
        self.eligible = []
        self.current_feasible_ope_set = []
        self.remaining_ope_list = []
        self.current_ope_mas_list = []
        self.opesWorkloadOnOneMa = None
        self.draw_gantt: bool = False
        self.jobs_available_list = None

        # load instance
        if data_source == 'case':
            lines, _, _ = case.get_case()  # Generate an instance and save it
        else:
            with open(case) as file_object:
                lines = file_object.readlines()
        self.num_jobs, self.num_mas, self.num_opes = nums_detec(lines)
        # load instance
        flag = 0
        matrix_proc_time = np.zeros((self.num_opes, self.num_mas))
        matrix_pre_proc = np.full((self.num_opes, self.num_opes), False)
        matrix_cal_cumul = np.zeros((self.num_opes, self.num_opes), dtype=int)
        nums_ope = []  # A list of the number of operations for each job
        opes_appertain = np.array([])
        num_ope_biases = []  # The id of the first operation of each job
        # Parse data line by line
        for line in lines:
            # first line
            if flag == 0:
                flag += 1
            # last line
            elif line == "\n":
                break
            # other
            else:
                num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
                num_ope_biases.append(num_ope_bias)
                # Detect information of this job and return the number of operations
                num_ope = self.edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
                nums_ope.append(num_ope)
                opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))
                flag += 1
        matrix_ope_ma_adj = np.where(matrix_proc_time > 0, 1, 0)

        self.proc_time = matrix_proc_time
        self.ope_ma = matrix_ope_ma_adj
        self.nums_ope_each = np.array(nums_ope, dtype=int)
        self.end_ope_biases = np.array(num_ope_biases, dtype=int) + self.nums_ope_each - 1
        self.num_ope_biases = num_ope_biases
        self.time = 0
        self.N: int = 0
        self.ope_step = copy.deepcopy(np.array(num_ope_biases, dtype=int))
        self.mask_job_procing = np.full(self.num_jobs, False)
        self.mask_job_finish = np.full(self.num_jobs, False)
        self.mask_mas_procing = np.full(self.num_mas, False)

        self.machines_info = np.zeros((self.num_mas, 3))
        self.machines_info[:, 0] = 1
        self.done = False

        self.schedule = np.full((self.num_opes, 4), float('inf'))

        self.old_proc_time = copy.deepcopy(self.proc_time)
        self.old_ope_step = copy.deepcopy(self.ope_step)
        self.old_ope_ma = copy.deepcopy(self.ope_ma)

        """
            in order to use more terminals, add more properties 
        """
        # the last processing time of operations to calculate the OWT
        self.jobs_proc_time = np.zeros(self.num_jobs)

    def edge_detec(self, line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
        '''
        Detect information of a job
        '''
        line_split = line.split()
        flag = 0
        flag_time = 0
        flag_new_ope = 1
        idx_ope = -1
        num_ope = 0  # Store the number of operations of this job
        num_option = np.array([])  # Store the number of processable machines for each operation of this job
        mac = 0
        for i in line_split:
            x = int(i)
            # The first number indicates the number of operations of this job
            if flag == 0:
                num_ope = x
                flag += 1
            # new operation detected
            elif flag == flag_new_ope:
                idx_ope += 1
                flag_new_ope += x * 2 + 1
                num_option = np.append(num_option, x)
                if idx_ope != num_ope - 1:
                    matrix_pre_proc[idx_ope + num_ope_bias][idx_ope + num_ope_bias + 1] = True
                if idx_ope != 0:
                    vector = np.zeros(matrix_cal_cumul.shape[0])
                    vector[idx_ope + num_ope_bias - 1] = 1
                    matrix_cal_cumul[:, idx_ope + num_ope_bias] = matrix_cal_cumul[:,
                                                                  idx_ope + num_ope_bias - 1] + vector
                flag += 1
            # not proc_time (machine)
            elif flag_time == 0:
                mac = x - 1
                flag += 1
                flag_time = 1
            # proc_time
            else:
                matrix_proc_time[idx_ope + num_ope_bias][mac] = x
                flag += 1
                flag_time = 0
        return num_ope

    def evlMakeSpan(self, individual, pset):
        # fitness values
        cal_priority = gp.compile(individual, pset)
        while not self.done:
            # get the action according to the func and environment
            ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)

            # get eligible machines and jobs
            op_proc_time = np.take_along_axis(self.proc_time, ope_step[:, None], axis=0)
            ma_eligible = np.broadcast_to(~self.mask_mas_procing, np.shape(op_proc_time))
            job_eligible = np.broadcast_to(~(self.mask_job_procing + self.mask_job_finish)[:, None],
                                           np.shape(op_proc_time))
            self.eligible = np.where(ma_eligible & job_eligible, op_proc_time, 0.0).T
            mask = self.eligible.flatten()
            action_feasible_actions = np.array(np.nonzero(mask)[0])
            fitness_list = []

            for i in range(len(action_feasible_actions)):
                self.get_action_index(action_feasible_actions[i], ope_step)
                self.opesWorkloadOnOneMa = self.eligible[self.current_ma]
                priority_value = cal_priority()
                fitness_list.append(priority_value)
            action_best_index = np.argmin(np.array(fitness_list))
            self.get_action_index(action_feasible_actions[action_best_index], ope_step)
            self.gp_step()
        make_span = copy.deepcopy(np.max(self.machines_info[:, 1]))
        validated_result = self.gp_validate()
        # if self.draw_gantt:
        #     self.gp_draw_gantt()
        self.gp_reset()
        if not validated_result:
            print("There is something wrong in the schedule")
        return make_span,

    """
    added by JPang to draw the gantt to see the schedule is feasible
    date:01/08/2023
    """

    # def gp_draw_gantt(self):
    #     plt.figure(figsize=(40, 15))
    #     plt.yticks(list(range(self.num_jobs)))
    #     colour_list = list(colours.cnames.keys())
    #     colour_list.remove('black')
    #     colour_list.remove('cyan')
    #     colour_idx = np.linspace(0, len(colours.cnames.keys()), num=self.num_jobs).astype(int)
    #     colour_list = [colour_list[i] for i in range(len(colour_idx))]
    #     ax = plt.gca()
    #     ax.spines[['right', 'top']].set_visible(False)
    #     schedule = copy.deepcopy(self.schedule).astype(int)
    #     for i in range(self.num_jobs):
    #         for j in range(self.nums_ope_each[i]):
    #             step = schedule[self.num_ope_biases[i] + j]
    #             plt.barh(step[1], width=step[3] - step[2], left=step[2], color=colour_list[i])
    #             plt.text(step[2] + (step[3] - step[2]) / 8, step[1], 'J%s\nO%s' % (i, j), color='k')
    #     plt.show()

    """
    added by JPang to make sure that the schedule is feasible 
    date: 31/07/2023 
    """

    def gp_validate(self):
        ma_gantt = [[] for _ in range(self.num_mas)]
        schedule = copy.deepcopy(self.schedule).astype(int)
        for i in range(self.num_opes):
            step = schedule[i, :]
            ma_gantt[step[1]].append([i, step[2], step[3]])
        proc_time = self.proc_time

        # check whether there are overlaps or wrong processing time on the machines
        flag_ma_overlap = 0
        flag_proc_time = 0
        for i in range(self.num_mas):
            ma_gantt[i].sort(key=lambda s: s[1])  # sorted by the start time
            for j in range(len(ma_gantt[i])):
                if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                    break
                if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:
                    flag_ma_overlap += 1
                if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0], i]:
                    flag_proc_time += 1

        # check job order and overlap
        flag_ope_overlap = 0
        for i in range(self.num_jobs):
            if self.nums_ope_each[i] <= 1:
                continue
            for j in range(self.nums_ope_each[i] - 1):
                step = self.schedule[self.num_ope_biases[i] + j]
                step_next = self.schedule[self.num_ope_biases[i] + j + 1]
                if step[3] > step_next[2]:
                    flag_ope_overlap += 1

        # check if all jobs are scheduled
        flag_unscheduled = 1 if np.isinf(self.schedule).any() else 0

        result = True if flag_proc_time + flag_ma_overlap + flag_ope_overlap + flag_unscheduled == 0 else False
        return result

    def gp_step(self):
        '''
        Environment transition function
        modified by JPang to remove unnecessary parts
        '''
        opes = self.current_ope
        mas = self.current_ma
        jobs = self.current_job
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = np.zeros(self.num_mas)
        remain_ope_ma_adj[mas] = 1
        self.ope_ma[opes, :] = remain_ope_ma_adj
        self.proc_time *= self.ope_ma

        # # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_time[opes, mas]

        self.machines_info[mas, 0] = 0
        self.machines_info[mas, 1] = self.time + proc_times
        # self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_info[mas, 2] = jobs

        self.jobs_proc_time[jobs] = self.time + proc_times

        # Update other variable according to actions
        self.ope_step[jobs] += 1
        self.mask_job_procing[jobs] = True
        self.mask_mas_procing[mas] = True
        self.mask_job_finish = np.where(self.ope_step == self.end_ope_biases + 1, True, self.mask_job_finish)
        self.done = self.mask_job_finish.all()

        # added by JPang, date: 31/07/2023, make sure that the schedule is feasible
        self.schedule[opes, 0] = 1
        self.schedule[opes, 1] = mas
        self.schedule[opes, 2] = self.time
        self.schedule[opes, 3] = self.time + proc_times

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while (flag_trans_2_next_time == 0) & (~self.done):
            self.gp_next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

    def gp_next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done)
        # available_time of machines
        a = self.machines_info[:, 1]

        # remain available_time greater than current time
        b = np.where(a > self.time, a, float('inf'))

        # Return the minimum value of available_time (the time to transit to)
        c = np.min(b)
        # Detect the machines that completed (at above time)
        d = np.where((a == c) & (self.machines_info[:, 0] == 0) & flag_need_trans, True, False)
        # The time for each batch to transit to or stay in
        e = flag_need_trans * c + ~flag_need_trans * self.time
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        self.machines_info[d, 0] = 1
        job_idxes = self.machines_info[:, 2][d].astype(int)

        self.mask_job_procing[job_idxes] = False
        self.mask_mas_procing[d] = False
        self.mask_job_finish = np.where(self.ope_step == self.end_ope_biases + 1, True, self.mask_job_finish)

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)
        op_proc_time = np.take_along_axis(self.proc_time, ope_step[:, None], axis=0)
        ma_eligible = np.broadcast_to(~self.mask_mas_procing, np.shape(op_proc_time))
        job_eligible = np.broadcast_to(~(self.mask_job_procing + self.mask_job_finish)[:, None], np.shape(op_proc_time))
        flag_trans_2_next_time = np.sum(np.where(ma_eligible & job_eligible, op_proc_time, 0.0))
        return flag_trans_2_next_time

    def get_action_index(self, action_indexes, ope_step):
        # Calculate the machine, job and operation index based on the action index
        self.current_ma = int(action_indexes / self.num_jobs)
        self.current_job = action_indexes % self.num_jobs
        self.current_ope = ope_step[self.current_job]

        self.remaining_ope_list = list(range(self.current_ope, self.end_ope_biases[self.current_job]))
        self.current_ope_mas_list = [idx for idx, value in enumerate(self.ope_ma[self.current_ope, :]) if value == 1]

        self.jobs_available_list = [idx for idx, value in enumerate(self.mask_job_procing) if value == False]
        ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)
        self.current_feasible_ope_set = ope_step[self.jobs_available_list]

    # get the processing time of current operations
    def get_PT(self):
        return self.proc_time[self.current_ope, self.current_ma]

    # get the waiting time of current operation since ready
    def get_OWT(self):
        return self.time - self.jobs_proc_time[self.current_job]

    # get the median processing time of the next operation
    # guess the NPT as the median processing time of current time considering that we have no idea of the next
    # operation on this machine because of the flexibility
    def get_NPT(self):
        return np.median(self.proc_time[self.current_ope, :])

    # get the work remaining for the job
    # use the median processing time of each operation
    def get_WKR(self):
        WKR = 0
        for ope_idx in self.remaining_ope_list:
            WKR += np.median(self.proc_time[ope_idx, :])
        return WKR

    # get the number of remaining operations of job
    def get_NOR(self):
        return len(self.remaining_ope_list)

    # get workload in the candidate operation set of the current machine
    def get_WIQ(self):
        return np.sum(self.opesWorkloadOnOneMa)

    # get the number of operations in the candidate operation set
    def get_NIQ(self):
        return np.count_nonzero(self.opesWorkloadOnOneMa)

    # get the number of candidate machines of the operations
    def get_NOS(self):
        return np.sum(self.ope_ma[self.current_ope, :])

    # get the machine waiting time
    def get_MWT(self):
        MWT = self.time - self.machines_info[self.current_ma, 1]
        return MWT

    """
        NO routing rule is used in this flexible scenario 
        so, the relative terminal is used to consider more information from other machines 
    """

    # get the relative processing time
    def get_rPT(self):
        pro_time = self.proc_time[self.current_ope, :]
        return pro_time[self.current_ma] - np.min(pro_time[pro_time != 0])

    # get the relative workload in the candidate operation set
    def get_rWIQ(self):
        opes_procing_time = self.proc_time[self.current_feasible_ope_set, :]
        mas_workload = np.zeros(self.num_mas)
        for idx_mas in range(self.num_mas):
            mas_procing_time = [i for i in opes_procing_time[:, idx_mas] if i != 0]
            if len(mas_procing_time) == 0:
                mas_workload[idx_mas] = np.inf
                continue
            mas_workload[idx_mas] = np.sum(mas_procing_time)
            # mas_procing_time_sorted = sorted(mas_procing_time)
            # if len(mas_procing_time_sorted) % 2 == 1:
            #     mas_workload[idx_mas] = mas_procing_time_sorted[len(mas_procing_time) // 2]
            # else:
            #     mas_workload[idx_mas] = (mas_procing_time_sorted[len(mas_procing_time) // 2 - 1] + mas_procing_time_sorted[len(mas_procing_time) // 2]) / 2
        return self.get_WIQ() - np.min(mas_workload)

    # get the relative number of operations in the candidate operation set
    def get_rNIQ(self):
        opes_mas = np.sum(self.ope_ma[self.current_feasible_ope_set, :], axis=0)
        return self.get_NIQ() - np.min(opes_mas[opes_mas != 0])

    # get the relative machine waiting time
    # def get_rMWT(self):
    #     MWT = self.get_MWT()
    #     mas_MWT = self.time - self.machines_info[[i for i in list(range(self.num_mas)) if i != self.current_ma], 1]
    #     mas_MWT = np.where(mas_MWT < 0, 0, mas_MWT)
    #     # if MWT <0:
    #     #     time.time()
    #     # # if np.count_nonzero(mas_MWT) > 1:
    #     # #     time.time()
    #     # if np.min(mas_MWT) > 0:
    #     #     time.time()
    #     return MWT - np.min(mas_MWT)
    #     # OWT = self.get_OWT()
    #     # opes_OWT = self.time - self.jobs_proc_time[self.jobs_available_list]
    #     # return OWT - np.min(opes_OWT)

    def gp_reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_time = copy.deepcopy(self.old_proc_time)
        self.ope_ma = copy.deepcopy(self.old_ope_ma)
        self.ope_step = copy.deepcopy(self.old_ope_step)

        self.time = 0
        self.N = 0
        self.mask_job_procing = np.full(self.num_jobs, False)
        self.mask_job_finish = np.full(self.num_jobs, False)
        self.mask_mas_procing = np.full(self.num_mas, False)
        self.schedule = np.full((self.num_opes, 4), float('inf'))

        self.machines_info = np.zeros((self.num_mas, 3))
        self.machines_info[:, 0] = 1
        self.done = False
        self.current_ma = 0
        self.current_job = 0
        self.current_ope = 0
        self.remaining_ope_list = []
        self.current_ope_mas_list = []
        self.eligible_opes = []
        self.eligible = []
        self.current_feasible_ope_set = []
        self.opesWorkloadOnOneMa = None
        self.draw_gantt = False

        self.jobs_proc_time = np.zeros(self.num_jobs)
        self.jobs_available_list = None

    # created by JPang to evaluate the piecewise gp
    def pwEvlMakeSpan(self, individual, pset, time):
        # fitness values
        if time is None:
            cal_priority = gp.compile(individual[0], pset)
        else:
            start_time = 0
            interval_width = time / (len(individual) + 1)
            time_slot = [start_time + i * interval_width for i in range(len(individual) + 1)]
        while not self.done:
            # get the action according to the func and environment
            ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)

            # get eligible machines and jobs
            op_proc_time = np.take_along_axis(self.proc_time, ope_step[:, None], axis=0)
            ma_eligible = np.broadcast_to(~self.mask_mas_procing, np.shape(op_proc_time))
            job_eligible = np.broadcast_to(~(self.mask_job_procing + self.mask_job_finish)[:, None],
                                           np.shape(op_proc_time))
            eligible = np.where(ma_eligible & job_eligible, op_proc_time, 0.0).T
            mask = eligible.flatten()
            action_feasible_actions = np.array(np.nonzero(mask)[0])
            fitness_list = []
            for i in range(len(action_feasible_actions)):
                self.get_action_index(action_feasible_actions[i], ope_step)
                self.next_ope = self.current_ope + 1 if self.current_ope + 1 < self.end_ope_biases[
                    self.current_job] else self.end_ope_biases[self.current_job]
                self.opesWorkloadOnOneMa = eligible[self.current_ma]
                if time is not None:
                    ind = next((individual[i] for i, t in enumerate(time_slot[0:-2]) if
                                t <= self.time < time_slot[i + 1]), individual[-1])
                    cal_priority = gp.compile(ind, pset)
                priority_value = cal_priority()
                fitness_list.append(priority_value)
            action_best_index = np.argmin(np.array(fitness_list))
            self.get_action_index(action_feasible_actions[action_best_index], ope_step)
            self.gp_step()
        make_span = copy.deepcopy(np.max(self.machines_info[:, 1]))
        validated_result = self.gp_validate()
        # if self.draw_gantt:
        #     self.gp_draw_gantt()
        self.gp_reset()
        if not validated_result:
            print("There is something wrong in the schedule")
        return make_span,

        #  rank characterised_rule_one_situation and reference_rule_one_situation

        # phenotypic_characterisation = []
        # while not self.done:
        #     # get the action according to the func and environment
        #     ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)
        #     # get eligible machines and jobs
        #     op_proc_time = np.take_along_axis(self.proc_time, ope_step[:, None], axis=0)
        #     ma_eligible = np.broadcast_to(~self.mask_mas_procing, np.shape(op_proc_time))
        #     job_eligible = np.broadcast_to(~(self.mask_job_procing + self.mask_job_finish)[:, None],
        #                                    np.shape(op_proc_time))
        #     self.eligible = np.where(ma_eligible & job_eligible, op_proc_time, 0.0).T
        #     mask = self.eligible.flatten()
        #     action_feasible_actions = np.array(np.nonzero(mask)[0])
        #     reference_rule_fitness_list = []
        #     characterised_rule_fitness_list = []
        #
        #     if len(action_feasible_actions) == min_queue_length:
        #         characterised_rule_cal_priority = gp.compile(individual, pset)
        #     for i in range(len(action_feasible_actions)):
        #         self.get_action_index(action_feasible_actions[i], ope_step)
        #         self.opesWorkloadOnOneMa = self.eligible[self.current_ma]
        #         if len(action_feasible_actions) == min_queue_length:
        #             characterised_rule_fitness_list.append(characterised_rule_cal_priority())
        #         priority_value = reference_rule_cal_priority()
        #         reference_rule_fitness_list.append(priority_value)
        #     action_best_index = np.argmin(np.array(reference_rule_fitness_list))
        #     if len(action_feasible_actions) == min_queue_length:
        #         characterised_rule_cal_best_idx = characterised_rule_fitness_list.index(
        #             min(characterised_rule_fitness_list))
        #         reference_rule_fitness_list_rank = np.argsort(np.argsort(reference_rule_fitness_list))
        #         phenotypic_characterisation.append(reference_rule_fitness_list_rank[characterised_rule_cal_best_idx])
        #
        #     self.get_action_index(action_feasible_actions[action_best_index], ope_step)
        #     self.gp_step()
        # self.gp_reset()


class StaticFlexibleEnv4PC(StaticFlexibleEnv):
    def __init__(self, case, data_source='case'):
        super().__init__(case, data_source)
        self.queue_count: int = 0
        self.situation_count: int = 0
        self.shuffle_seed = 8295342
        self.algo_seed: int = int(sys.argv[3])
        self.situations: list[dict] = []
        self.num_decision_situation: int = 30  # in dynamic situation, the number is 20
        self.min_queue_length: int = 3  # in dynamic situation, there is no min queue
        self.max_queue_length: int = 8  # in dynamic situation, the number is 7

    # set the reference rule as WIQ
    def reference_rule(self):
        return self.proc_time[self.current_ope, self.current_ma]
        # return np.sum(self.opesWorkloadOnOneMa)

    # get the situation for calculating PC with WIQ as the default rule
    def get_situation_for_calculate_pc(self, pset: gp.PrimitiveSet):
        terminal_set_name = [terminal.name for terminal in pset.terminals[object]]

        while not self.done:
            # get the action according to the func and environment
            ope_step = np.where(self.ope_step > self.end_ope_biases, self.end_ope_biases, self.ope_step)
            # get eligible machines and jobs
            op_proc_time = np.take_along_axis(self.proc_time, ope_step[:, None], axis=0)
            ma_eligible = np.broadcast_to(~self.mask_mas_procing, np.shape(op_proc_time))
            job_eligible = np.broadcast_to(~(self.mask_job_procing + self.mask_job_finish)[:, None],
                                           np.shape(op_proc_time))
            self.eligible = np.where(ma_eligible & job_eligible, op_proc_time, 0.0).T
            mask = self.eligible.flatten()
            action_feasible_actions = np.array(np.nonzero(mask)[0])
            reference_rule_fitness_list: list[np.ndarray] = []

            if len(action_feasible_actions) >= self.min_queue_length and len(action_feasible_actions) <= self.max_queue_length:
                record_as_situation_for_pc: bool = True
                one_situation_for_pc: dict[str, list] = defaultdict(list)
            else:
                record_as_situation_for_pc: bool = False

            for i in range(len(action_feasible_actions)):
                self.get_action_index(action_feasible_actions[i], ope_step)
                self.opesWorkloadOnOneMa = self.eligible[self.current_ma]
                if record_as_situation_for_pc:
                    for terminal in terminal_set_name:
                        one_situation_for_pc[terminal].append(eval("self."+re.split("_", terminal)[1])())

                priority_value = self.reference_rule()
                reference_rule_fitness_list.append(priority_value)

            if record_as_situation_for_pc:
                one_situation_for_pc["reference rule"] = reference_rule_fitness_list
                self.situations.append(one_situation_for_pc)

            action_best_index = np.argmin(np.array(reference_rule_fitness_list))

            self.get_action_index(action_feasible_actions[action_best_index], ope_step)
            self.gp_step()
        self.gp_reset()

        reset_seed(self.shuffle_seed)
        non_shuffle_situations_for_PC = self.situations
        self.num_decision_situation = min(self.num_decision_situation, len(non_shuffle_situations_for_PC))
        shuffle_idx = random.sample(list(range(0, len(non_shuffle_situations_for_PC))), self.num_decision_situation)
        self.situations = [non_shuffle_situations_for_PC[idx] for idx in shuffle_idx]
        reset_seed(self.algo_seed)

    def get_pc(self, individual, pset):
        characterised_rule_cal_priority = gp.compile(individual, pset)
        characterised_rule_pc: list[np.ndarray] = []
        for situation_count, one_situation in enumerate(self.situations):
            self.situation_count = situation_count
            reference_rule_one_situation: list[np.ndarray] = one_situation["reference rule"]
            characterised_rule_one_situation: list[np.ndarray] = []
            for queue_count in range(len(one_situation["reference rule"])):
                self.queue_count = queue_count
                characterised_rule_one_situation.append(characterised_rule_cal_priority())
            characterised_rule_best_idx = characterised_rule_one_situation.index(min(characterised_rule_one_situation))
            reference_rule_one_situation_rank = np.argsort(np.argsort(reference_rule_one_situation))
            characterised_rule_pc.append(reference_rule_one_situation_rank[characterised_rule_best_idx])
        return characterised_rule_pc

    def get_PT(self):
        return self.situations[self.situation_count]["get_PT"][self.queue_count]

    # get the waiting time of current operation since ready
    def get_OWT(self):
        return self.situations[self.situation_count]["get_OWT"][self.queue_count]

    # get the median processing time of the next operation
    # guess the NPT as the median processing time of current time considering that we have no idea of the next
    # operation on this machine because of the flexibility
    def get_NPT(self):
        return self.situations[self.situation_count]["get_NPT"][self.queue_count]

    # get the work remaining for the job
    # use the median processing time of each operation
    def get_WKR(self):
        return self.situations[self.situation_count]["get_WKR"][self.queue_count]

    # get the number of remaining operations of job
    def get_NOR(self):
        return self.situations[self.situation_count]["get_NOR"][self.queue_count]

    # get workload in the candidate operation set of the current machine
    def get_WIQ(self):
        return self.situations[self.situation_count]["get_WIQ"][self.queue_count]

    # get the number of operations in the candidate operation set
    def get_NIQ(self):
        return self.situations[self.situation_count]["get_NIQ"][self.queue_count]

    # get the number of candidate machines of the operations
    def get_NOS(self):
        return self.situations[self.situation_count]["get_NOS"][self.queue_count]

    # get the machine waiting time
    def get_MWT(self):
        return self.situations[self.situation_count]["get_MWT"][self.queue_count]

    # get the relative processing time
    def get_rPT(self):
        return self.situations[self.situation_count]["get_rPT"][self.queue_count]

    # get the relative workload in the candidate operation set
    def get_rWIQ(self):
        return self.situations[self.situation_count]["get_rWIQ"][self.queue_count]

    # get the relative number of operations in the candidate operation set
    def get_rNIQ(self):
        return self.situations[self.situation_count]["get_rNIQ"][self.queue_count]

    # created for the situation generating
    def PT(self):
        return self.proc_time[self.current_ope, self.current_ma]

    # get the waiting time of current operation since ready
    def OWT(self):
        return self.time - self.jobs_proc_time[self.current_job]

    # get the median processing time of the next operation
    # guess the NPT as the median processing time of current time considering that we have no idea of the next
    # operation on this machine because of the flexibility
    def NPT(self):
        return np.median(self.proc_time[self.current_ope, :])

    # get the work remaining for the job
    # use the median processing time of each operation
    def WKR(self):
        WKR = 0
        for ope_idx in self.remaining_ope_list:
            WKR += np.median(self.proc_time[ope_idx, :])
        return WKR

    # get the number of remaining operations of job
    def NOR(self):
        return len(self.remaining_ope_list)

    # get workload in the candidate operation set of the current machine
    def WIQ(self):
        return np.sum(self.opesWorkloadOnOneMa)

    # get the number of operations in the candidate operation set
    def NIQ(self):
        return np.count_nonzero(self.opesWorkloadOnOneMa)

    # get the number of candidate machines of the operations
    def NOS(self):
        return np.sum(self.ope_ma[self.current_ope, :])

    # get the machine waiting time
    def MWT(self):
        MWT = self.time - self.machines_info[self.current_ma, 1]
        return MWT

    """
        NO routing rule is used in this flexible scenario 
        so, the relative terminal is used to consider more information from other machines 
    """

    # get the relative processing time
    def rPT(self):
        pro_time = self.proc_time[self.current_ope, :]
        return pro_time[self.current_ma] - np.min(pro_time[pro_time != 0])

    # get the relative workload in the candidate operation set
    def rWIQ(self):
        opes_procing_time = self.proc_time[self.current_feasible_ope_set, :]
        mas_workload = np.zeros(self.num_mas)
        for idx_mas in range(self.num_mas):
            mas_procing_time = [i for i in opes_procing_time[:, idx_mas] if i != 0]
            if len(mas_procing_time) == 0:
                mas_workload[idx_mas] = np.inf
                continue
            mas_workload[idx_mas] = np.sum(mas_procing_time)
            # mas_procing_time_sorted = sorted(mas_procing_time)
            # if len(mas_procing_time_sorted) % 2 == 1:
            #     mas_workload[idx_mas] = mas_procing_time_sorted[len(mas_procing_time) // 2]
            # else:
            #     mas_workload[idx_mas] = (mas_procing_time_sorted[len(mas_procing_time) // 2 - 1] + mas_procing_time_sorted[len(mas_procing_time) // 2]) / 2
        return self.WIQ() - np.min(mas_workload)

    # get the relative number of operations in the candidate operation set
    def rNIQ(self):
        opes_mas = np.sum(self.ope_ma[self.current_feasible_ope_set, :], axis=0)
        return self.NIQ() - np.min(opes_mas[opes_mas != 0])
