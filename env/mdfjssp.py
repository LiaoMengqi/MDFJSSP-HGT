import gymnasium as gym

from env.base_class import State, Action
from utils.case_generator import CaseGenerator, Case
import torch


class MDFJSSPEnv(gym.Env):
    def __init__(self, args, cases: list[Case], device='gpu'):
        self.case_generator = CaseGenerator(args['num_machine'], args['num_job'], args['num_operation_max'],
                                            dynamic_prop=args['dynamic_prop'],
                                            sigma_pres_time=args['sigma_pres_time'],
                                            mu_lbd=args['mu_lbd'])
        self.num_machine = args['num_machine']
        self.num_operation_max = args['num_operation_max']
        self.num_cases = args['num_cases']
        self.d_operation_raw = args['d_operation_raw']
        self.d_machine_raw = args['d_machine_raw']
        self.d_arc_raw = args['d_arc_raw']
        self.device = device

        # cases for evaluation
        self.cases = cases
        if self.cases is None:
            # generate cases for train
            self.cases = self.case_generator.generate_cases(self.num_cases)

        self.cur_time = 0
        self.state = None
        self.suc_job_list = None
        self.break_down_list = None
        self.machine_broken_timer = None
        self.action_timer = None
        self.case_finish_earliest = None
        self.machine_free_time = None
        self.machine_free_time_recoder = None

    def reset(self, keep_cases=True, seed=None) -> State:
        self.suc_job_list = []
        self.break_down_list = []
        self.cur_time = 0

        # initialize state
        if keep_cases:
            if self.cases is None:
                self.cases = self.case_generator.generate_cases(self.num_cases)
            self.num_operation_max = max([case.num_operation + case.num_operation_suc for case in self.cases]) + 1
            self.num_cases = len(self.cases)
            self.state = self.init_state(self.cases)

        else:
            self.cases = self.case_generator.generate_cases(self.num_cases)
            self.num_operation_max = max([case.num_operation + case.num_operation_suc for case in self.cases]) + 1
            self.state = self.init_state(self.cases)
        self.machine_broken_timer = torch.zeros((self.num_cases, self.num_machine, 1), dtype=torch.long).to(self.device)
        self.action_timer = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max),
                                        dtype=torch.long).to(self.device)
        # update features
        self.state.update_feature(self.action_timer, self.cur_time)
        self.case_finish_earliest = self.state.operation_finished_time_earliest.max(dim=-1)[0]
        self.machine_free_time = torch.zeros((self.num_cases, self.num_machine), dtype=torch.long, device=self.device)
        self.machine_free_time_recoder = torch.zeros((self.num_cases), dtype=torch.long, device=self.device)
        return self.state

    def init_state(self, cases) -> State:
        self.num_machine = cases[0].num_machine
        # initialize mask
        mask_machine = torch.ones((self.num_cases, self.num_machine, 1), dtype=torch.bool, device=self.device)
        mask_broken_down = torch.zeros((self.num_cases, self.num_machine, 1), dtype=torch.bool, device=self.device)

        mask_operation = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool, device=self.device)
        mask_operation_finished = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool,
                                              device=self.device)
        mask_operation_finished[:, :, 0] = True
        mask_operation_processing = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.bool,
                                                device=self.device)
        mask_machine_operation = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max),
                                             dtype=torch.bool, device=self.device)
        # initialize feature
        operation_raw_feature = torch.zeros(self.num_cases, self.num_operation_max, self.d_operation_raw,
                                            device=self.device)
        machine_raw_feature = torch.zeros(self.num_cases, self.num_machine, self.d_machine_raw, device=self.device)
        arc_raw_feature = torch.zeros(self.num_cases, self.num_machine, self.num_operation_max, self.d_arc_raw,
                                      device=self.device)
        # other information
        accu_matrix = torch.zeros((self.num_cases, self.num_operation_max, self.num_operation_max),
                                  dtype=torch.long, device=self.device)
        operation_adj_matrix = torch.zeros((self.num_cases, self.num_operation_max, self.num_operation_max),
                                           dtype=torch.bool, device=self.device)
        operation_time = torch.zeros((self.num_cases, self.num_machine, self.num_operation_max), dtype=torch.long,
                                     device=self.device)
        last_operation_per_job = torch.zeros((self.num_cases, 1, self.num_operation_max), dtype=torch.long,
                                             device=self.device)
        machine_broken_record = torch.zeros(self.num_cases, self.num_machine, 1, device=self.device)

        for i in range(self.num_cases):
            operation_bias = 1
            for j, job in enumerate(cases[i].init_job_list):
                last_operation_per_job[i, :,
                operation_bias:operation_bias + len(job.operation_list)] = operation_bias + len(job.operation_list) - 1
                for k, operation in enumerate(job.operation_list):
                    # feature
                    operation_raw_feature[
                        i, operation_bias + k, 2] = (k + 1) / job.num_operation  # progress of an operation in a job
                    operation_raw_feature[i, operation_bias + k, 5] = len(operation.available_machine)
                    operation_raw_feature[i, operation_bias + k, 6] = len(job.operation_list) - k - 1
                    # mask
                    mask_machine_operation[i, operation.available_machine, operation_bias + k] = True
                    operation_time[i, operation.available_machine, operation_bias + k] = torch.LongTensor(
                        operation.operation_times).to(self.device)  # operation time on different machine
                    mask_operation[i, 0, operation_bias + k] = True
                    operation_adj_matrix[i, 0 if k == 0 else operation_bias + k - 1, operation_bias + k] = True
                    # other
                    accu_matrix[i, operation_bias:operation_bias + k + 1, operation_bias + k] = 1
                operation_bias += cases[i].init_job_list[j].num_operation
            # success job initialization
            suc_job_list = sorted(cases[i].suc_job_list, key=lambda x: x[1])
            suc_job_list_case = []
            for job, arrive_time in suc_job_list:
                adj_matrix = torch.zeros(1, self.num_machine, job.num_operation, dtype=torch.long, device=self.device)
                for index, operation in enumerate(job.operation_list):
                    adj_matrix[0, operation.available_machine, index] = torch.LongTensor(
                        operation.operation_times).to(self.device)
                suc_job_list_case.append([adj_matrix, arrive_time])
            self.suc_job_list.append(suc_job_list_case)
            # break down initialization
            broken_down_list = sorted(cases[i].broken_down_list, key=lambda x: x[1])
            self.break_down_list.append(broken_down_list)

        temp = operation_time.clone()
        temp[temp == 0] = torch.iinfo(operation_time.dtype).max
        temp = (temp.min(dim=1)[0]).unsqueeze(1)
        operation_finished_time_earliest = (temp.float() @ accu_matrix.float()).long()
        # initialize state
        state = State(operation_raw_feature=operation_raw_feature, machine_raw_feature=machine_raw_feature,
                      arc_raw_feature=arc_raw_feature, mask_broken_down=mask_broken_down,
                      mask_machine=mask_machine, mask_machine_operation=mask_machine_operation,
                      mask_operation=mask_operation, mask_operation_processing=mask_operation_processing,
                      mask_operation_finished=mask_operation_finished, operation_adj_matrix=operation_adj_matrix,
                      operation_time=operation_time, accu_matrix=accu_matrix,
                      machine_broken_record=machine_broken_record,
                      operation_finished_time_earliest=operation_finished_time_earliest,
                      last_operation_per_job=last_operation_per_job)
        return state

    def process(self):
        mask_machine = torch.all(self.action_timer == 0, dim=2, keepdim=True)
        mask_machine |= self.state.mask_machine
        mask_operation_processing = ~torch.all(self.action_timer == 0, dim=1, keepdim=True)
        mask_operation_finished = (
                                          mask_operation_processing ^ self.state.mask_operation_processing
                                  ) | self.state.mask_operation_finished
        mask_operation_processing &= self.state.mask_operation_processing
        self.state.update(mask_operation_processing=mask_operation_processing, mask_machine=mask_machine,
                          mask_operation_finished=mask_operation_finished)

    def handle_new_job(self):
        mask_operation = self.state.mask_operation
        indices = torch.arange(mask_operation.size(2)).expand_as(mask_operation).to(self.device)
        second_false_indices = (mask_operation == False).int().cumsum(dim=2).eq(2) & (mask_operation == False)
        index = torch.where(second_false_indices, indices, torch.full_like(indices, mask_operation.size(2))).min(dim=2)[
            0]
        operation_time = self.state.operation_time
        mask_machine_operation = self.state.mask_machine_operation
        operation_adj_matrix = self.state.operation_adj_matrix
        accu_matrix = self.state.accu_matrix
        operation_finished_time_earliest = self.state.operation_finished_time_earliest
        last_operation_per_job = self.state.last_operation_per_job

        operation_raw_feature = self.state.operation_raw_feature
        for i in range(self.num_cases):
            remove_list = []
            for item in self.suc_job_list[i]:
                adj_matrix, arrive_time = item
                if arrive_time <= self.cur_time:
                    remove_list.append(item)
                    num_operation = adj_matrix.size(2)
                    # operation rwa feature
                    operation_raw_feature[i, index[i]:index[i] + num_operation, 1] = (
                            adj_matrix.sum(dim=1).float() / (adj_matrix != 0).sum(dim=1).float()).squeeze()
                    operation_raw_feature[i, index[i]:index[i] + num_operation, 2] = (
                            (torch.arange(num_operation, dtype=torch.float) + 1) / num_operation)
                    num_ava_machine = (adj_matrix > 0).float().sum(dim=1).squeeze()
                    operation_raw_feature[i, index[i]:index[i] + num_operation, 5] = num_ava_machine
                    operation_raw_feature[i, index[i]:index[i] + num_operation, 6] = torch.arange(num_operation - 1, -1,
                                                                                                  -1)

                    mask_operation[i, 0, index[i]:index[i] + num_operation] = True
                    operation_adj_matrix[i, 0, index[i]] = True
                    v = torch.arange(index[i].item(), index[i].item() + num_operation - 1)
                    u = v + 1
                    operation_adj_matrix[i, v, u] = True
                    operation_time[i, :, index[i]:index[i] + num_operation] = adj_matrix
                    mask_machine_operation = mask_machine_operation + (operation_time != 0)
                    matrix = torch.ones(num_operation, num_operation, dtype=torch.long, device=self.device)
                    matrix = torch.triu(matrix)
                    adj_matrix[adj_matrix == 0] = torch.iinfo(operation_time.dtype).max
                    time_finished = adj_matrix.min(dim=1)[0].unsqueeze(1)
                    time_finished = (time_finished.float() @ matrix.float()).long() + self.cur_time

                    # operation finished time
                    operation_finished_time_earliest[i, :, index[i]:index[i] + num_operation] = time_finished.squeeze(1)
                    accu_matrix[i, index[i]:index[i] + num_operation, index[i]:index[i] + num_operation] = matrix

                    last_operation_per_job[i, :, index[i]:index[i] + num_operation] = index[i] + num_operation - 1
                    index[i] = index[i] + num_operation
                else:
                    break
            for item in remove_list:
                self.suc_job_list[i].remove(item)

        self.state.update(mask_operation=mask_operation, operation_time=operation_time,
                          mask_machine_operation=mask_machine_operation, operation_adj_matrix=operation_adj_matrix,
                          accu_matrix=accu_matrix, operation_finished_time_earliest=operation_finished_time_earliest,
                          last_operation_per_job=last_operation_per_job)

    def handle_break_down(self):
        self.state.machine_broken_record[self.state.mask_broken_down] += 1
        # # check break down
        for i in range(self.num_cases):
            remove_list = []
            for break_down in self.break_down_list[i]:
                machine_id, occur_time, time_for_repair = break_down
                if occur_time <= self.cur_time:
                    remove_list.append(break_down)
                    self.machine_broken_timer[i, machine_id, :] = time_for_repair
                    self.state.mask_broken_down[i, machine_id, :] = True
                    # self.state.machine_raw_feature[i, machine_id, 4] += 1
                else:
                    break
            for break_down in remove_list:
                self.break_down_list[i].remove(break_down)
        mask_broken_down = self.machine_broken_timer != 0

        # interrupt
        action_mask = self.action_timer != 0
        mask_operation_stop = ~(
                action_mask.transpose(1, 2).float() @ mask_broken_down.float()
        ).bool().transpose(1, 2)
        mask_operation_processing = self.state.mask_operation_processing * mask_operation_stop
        self.action_timer.masked_fill_(mask_broken_down.expand_as(self.action_timer), 0)
        mask_machine = self.state.mask_machine | mask_broken_down
        # update state
        self.state.update(mask_broken_down=mask_broken_down, mask_operation_processing=mask_operation_processing,
                          mask_machine=mask_machine)

    def blocked_operation(self):
        mask = self.state.mask_operation_finished.clone()
        mask[:, :, 1:].copy_(mask[:, :, :-1])
        mask_unfinished = self.state.mask_operation & ~(
                self.state.mask_operation_finished | self.state.mask_operation_processing
        )
        mask *= mask_unfinished
        return torch.any(mask, dim=-1)

    def step(self, actions: Action):
        # update state mask according to action
        mask_operation_processing = torch.zeros(self.num_cases, 1, self.num_operation_max).bool().to(self.device)
        mask_operation_processing[actions.runnable_cases, :, actions.actions[:, 1][actions.runnable_cases]] = True
        mask_operation_processing = mask_operation_processing + self.state.mask_operation_processing
        mask_machine = torch.ones(self.num_cases, self.num_machine, 1).bool().to(self.device)
        mask_machine[actions.runnable_cases, actions.actions[:, 0][actions.runnable_cases], :] = False
        mask_machine = mask_machine * self.state.mask_machine
        self.action_timer[actions.runnable_cases, actions.actions[:, 0][actions.runnable_cases],
        actions.actions[:, 1][actions.runnable_cases]] = self.state.operation_time[
            actions.runnable_cases, actions.actions[:, 0][actions.runnable_cases],
            actions.actions[:, 1][actions.runnable_cases]]
        self.state.update(mask_operation_processing=mask_operation_processing, mask_machine=mask_machine)

        # calculate finish time reward
        self.state.update_finish_time(self.action_timer, self.cur_time, actions.runnable_cases)
        new_finish_time = self.state.operation_finished_time_earliest.max(dim=-1)[0]
        finish_time_reward = self.case_finish_earliest - new_finish_time
        self.case_finish_earliest = new_finish_time

        # calculate free time reward
        free_time_reward = torch.zeros(self.num_cases, device=self.device, dtype=torch.long)
        new_record = self.machine_free_time.sum(dim=-1)[actions.runnable_cases]
        free_time_reward[actions.runnable_cases] = self.machine_free_time_recoder[actions.runnable_cases] - new_record
        self.machine_free_time_recoder[actions.runnable_cases] = new_record
        reward = torch.cat([finish_time_reward, free_time_reward.unsqueeze(-1)], dim=-1)

        # loop if no new action could be taken and exist job have not been finished
        runnable, runnable_cases, finished_batch, finished = self.state.check_state()
        while (not runnable) and (not finished):
            self.cur_time = self.cur_time + 1
            self.action_timer = torch.clamp(torch.sub(self.action_timer, 1), min=0)
            self.machine_broken_timer = torch.clamp(torch.sub(self.machine_broken_timer, 1), min=0)
            self.process()
            self.handle_new_job()
            self.handle_break_down()
            # record free time of machine
            mask = (self.state.mask_machine & self.state.mask_broken_down).squeeze(-1)
            is_blocked = self.blocked_operation()  # still exits blocked operation
            mask = mask * is_blocked
            self.machine_free_time[mask] = self.machine_free_time[mask] + 1
            runnable, runnable_cases, finished_batch, finished = self.state.check_state()
        self.state.update_feature(self.action_timer, self.cur_time, runnable_cases)

        return self.state, reward, finished, finished, None

    def render(self):
        pass
