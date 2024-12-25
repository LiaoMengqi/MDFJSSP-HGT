import torch
import copy
from typing import Optional


class State:
    def __init__(self,
                 operation_raw_feature: torch.Tensor,
                 machine_raw_feature: torch.Tensor,
                 arc_raw_feature: torch.Tensor,
                 mask_broken_down: torch.Tensor,
                 mask_machine: torch.Tensor,
                 mask_machine_operation: torch.Tensor,
                 mask_operation: torch.Tensor,
                 mask_operation_processing: torch.Tensor,
                 mask_operation_finished: torch.Tensor,
                 operation_adj_matrix: torch.Tensor,
                 operation_time: Optional[torch.Tensor],
                 accu_matrix: Optional[torch.Tensor],
                 machine_broken_record: Optional[torch.Tensor],
                 operation_finished_time_earliest: Optional[torch.Tensor],
                 last_operation_per_job: Optional[torch.Tensor]):
        super(State, self).__init__()
        # feature
        self.operation_raw_feature = operation_raw_feature
        self.machine_raw_feature = machine_raw_feature
        self.arc_raw_feature = arc_raw_feature
        # mask
        self.mask_broken_down = mask_broken_down
        self.mask_machine = mask_machine

        self.mask_machine_operation = mask_machine_operation

        self.mask_operation = mask_operation
        self.mask_operation_processing = mask_operation_processing
        self.mask_operation_finished = mask_operation_finished

        self.operation_adj_matrix = operation_adj_matrix

        # other
        self.operation_time = operation_time
        self.accu_matrix = accu_matrix
        self.machine_broken_record = machine_broken_record
        self.operation_finished_time_earliest = operation_finished_time_earliest
        self.last_operation_per_job = last_operation_per_job

        # static feature
        if self.operation_time is not None:
            self.operation_raw_feature[:, :, 1] = ((self.operation_time.sum(dim=1)).float() / (
                    self.mask_machine_operation.sum(dim=1).float() + 1e-5))

    def to(self, device):
        self.operation_raw_feature = self.operation_raw_feature.to(device)
        self.machine_raw_feature = self.machine_raw_feature.to(device)
        self.arc_raw_feature = self.arc_raw_feature.to(device)
        self.mask_broken_down = self.mask_broken_down.to(device)
        self.mask_machine = self.mask_machine.to(device)
        self.mask_machine_operation = self.mask_machine_operation.to(device)
        self.mask_operation = self.mask_operation.to(device)
        self.mask_operation_processing = self.mask_operation_processing.to(device)
        self.mask_operation_finished = self.mask_operation_finished.to(device)
        self.operation_adj_matrix = self.operation_adj_matrix.to(device)
        if self.operation_time is not None:
            self.operation_time = self.operation_time.to(device)
        if self.accu_matrix is not None:
            self.accu_matrix = self.accu_matrix.to(device)
        if self.machine_broken_record is not None:
            self.machine_broken_record = self.machine_broken_record.to(device)
        if self.operation_finished_time_earliest is not None:
            self.operation_finished_time_earliest = self.operation_finished_time_earliest.to(device)
        if self.last_operation_per_job is not None:
            self.last_operation_per_job = self.last_operation_per_job.to(device)
        return self

    def update(self,
               operation_raw_feature=None,
               machine_raw_feature=None,
               arc_raw_feature=None,
               mask_broken_down=None,
               mask_machine=None,
               mask_machine_operation=None,
               mask_operation=None,
               mask_operation_processing=None,
               mask_operation_finished=None,
               operation_adj_matrix=None,
               operation_time=None,
               accu_matrix=None,
               machine_broken_record=None,
               operation_finished_time_earliest=None,
               last_operation_per_job=None):
        if operation_raw_feature is not None:
            self.operation_raw_feature = operation_raw_feature
        if machine_raw_feature is not None:
            self.machine_raw_feature = machine_raw_feature
        if arc_raw_feature is not None:
            self.arc_raw_feature = arc_raw_feature
        if mask_broken_down is not None:
            self.mask_broken_down = mask_broken_down
        if mask_machine is not None:
            self.mask_machine = mask_machine
        if mask_machine_operation is not None:
            self.mask_machine_operation = mask_machine_operation
        if mask_operation is not None:
            self.mask_operation = mask_operation
        if mask_operation_processing is not None:
            self.mask_operation_processing = mask_operation_processing
        if mask_operation_finished is not None:
            self.mask_operation_finished = mask_operation_finished
        if operation_adj_matrix is not None:
            self.operation_adj_matrix = operation_adj_matrix
        if operation_time is not None:
            self.operation_time = operation_time
        if accu_matrix is not None:
            self.accu_matrix = accu_matrix
        if machine_broken_record is not None:
            self.machine_broken_record = machine_broken_record
        if operation_finished_time_earliest is not None:
            self.operation_finished_time_earliest = operation_finished_time_earliest
        if last_operation_per_job is not None:
            self.last_operation_per_job = last_operation_per_job

    def update_finish_time(self, action_timer: torch.Tensor, time_now, runnable_cases=None):
        # estimated finish time
        temp = self.operation_time.clone()
        processing = self.mask_operation_processing.transpose(-1, -2).squeeze(-1)
        finished = self.mask_operation_finished.transpose(-1, -2).squeeze(-1)

        if processing.any():  # keep the processing operation time
            x = action_timer.transpose(-1, -2)[processing].max(dim=-1)[0] + time_now
            temp.transpose(-1, -2)[processing] = x.unsqueeze(-1)
        temp[temp == 0] = torch.iinfo(self.operation_time.dtype).max
        operation_min_time = temp.min(dim=1, keepdim=True)[0]

        if finished.any():  # keep the finished operation time
            operation_min_time.transpose(-1, -2)[finished] = 0

        matrix = self.accu_matrix
        finished_time_processing_start = (operation_min_time.float() @ matrix.float()).long()

        temp = self.operation_time.clone()
        temp[temp == 0] = torch.iinfo(self.operation_time.dtype).max
        operation_min_time = temp.min(dim=1, keepdim=True)[0]

        scheduled = processing | finished
        operation_min_time.transpose(-1, -2)[scheduled] = 0

        # operation_min_time = operation_min_time.transpose(-1, -2)
        finished_time_unscheduled = (operation_min_time.float() @ matrix.float()).long() + time_now
        finished_time_unscheduled.transpose(-1, -2)[scheduled] = 0

        operation_finished_time_earliest = torch.maximum(finished_time_processing_start,
                                                         finished_time_unscheduled)
        mask_unschedule = self.mask_operation & (~self.mask_operation_finished)
        if runnable_cases is None:
            self.operation_finished_time_earliest[mask_unschedule] = operation_finished_time_earliest[mask_unschedule]
        else:
            mask_unschedule[~runnable_cases] = False
            self.operation_finished_time_earliest[mask_unschedule] = operation_finished_time_earliest[mask_unschedule]

    def update_feature(self, action_timer: torch.Tensor, time_now, runnable_cases=None):
        # operation status
        self.operation_raw_feature[:, :, 0].masked_fill_(self.mask_operation.squeeze(1), 0)
        self.operation_raw_feature[:, :, 0].masked_fill_(self.mask_operation_processing.squeeze(1), 1)
        # processed time left
        self.operation_raw_feature[:, :, 3] = action_timer.sum(dim=1)

        # estimated finish time
        self.update_finish_time(action_timer, time_now, runnable_cases)

        self.operation_raw_feature[:, :, 4] = torch.gather(self.operation_finished_time_earliest, dim=2,
                                                           index=self.last_operation_per_job).squeeze(1)

        # machine status
        self.machine_raw_feature[:, :, 0].masked_fill_(self.mask_machine.squeeze(-1), 0)
        self.machine_raw_feature[:, :, 0].masked_fill_(~self.mask_machine.squeeze(-1), 1)
        self.machine_raw_feature[:, :, 0].masked_fill_(self.mask_broken_down.squeeze(-1), -1)
        # operation connected to machine
        self.machine_raw_feature[:, :, 1] = (self.mask_machine_operation * (~self.mask_operation_finished) * (
            self.mask_operation)).sum(dim=-1).float()
        temp = self.operation_time.clone()
        temp[action_timer == 0] = 0
        self.machine_raw_feature[:, :, 2] = (temp - action_timer).sum(dim=-1).float()
        self.machine_raw_feature[:, :, 3] = (self.machine_broken_record / (time_now + 1e-5)).squeeze(-1)

        # arc
        self.arc_raw_feature[:, :, :, 0] = self.operation_time.float()
        max_value = torch.max(self.arc_raw_feature[:, :, :, 0])
        self.arc_raw_feature[:, :, :, 0] = torch.where(self.arc_raw_feature[:, :, :, 0] == 0,
                                                       max_value,
                                                       self.arc_raw_feature[:, :, :, 0])
        self.arc_raw_feature[:, :, :, 1] = (self.operation_time.float() - action_timer.float()) * torch.sign(
            action_timer.float())

    def get_machine_encoder_mask(self):
        return self.mask_machine_operation * self.mask_operation * ~self.mask_operation_finished

    def get_operation_encoder_mask(self):
        return self.mask_machine_operation

    def get_action_mask(self):
        mask_available = self.mask_operation_finished.float() @ self.operation_adj_matrix.float()
        mask_available = mask_available.bool()
        action_mask = (self.mask_machine * ~self.mask_broken_down * self.mask_machine_operation *
                       self.mask_operation * ~self.mask_operation_finished * mask_available *
                       ~self.mask_operation_processing)
        runnable_cases = action_mask.any(dim=-1).any(dim=-1)
        return action_mask, runnable_cases

    def check_state(self):
        # runnable
        mask, runnable_cases = self.get_action_mask()
        runnable = torch.any(runnable_cases).item()
        # finished
        finished_batch = ~((~self.mask_operation_finished * self.mask_operation).any(dim=-1).any(dim=-1))
        finished = (torch.all(finished_batch)).item()
        return runnable, runnable_cases, finished_batch, finished

    @staticmethod
    def package(states: list, index):
        operation_raw_feature = torch.cat(
            [state.operation_raw_feature.unsqueeze(0) for state in states], dim=0)[index]
        machine_raw_feature = torch.cat([state.machine_raw_feature.unsqueeze(0) for state in states], dim=0)[index]
        arc_raw_feature = torch.cat([state.arc_raw_feature.unsqueeze(0) for state in states], dim=0)[index]
        mask_broken_down = torch.cat([state.mask_broken_down.unsqueeze(0) for state in states], dim=0)[index]
        mask_machine = torch.cat([state.mask_machine.unsqueeze(0) for state in states], dim=0)[index]
        mask_machine_operation = torch.cat([state.mask_machine_operation.unsqueeze(0) for state in states], dim=0)[
            index]
        mask_operation = torch.cat([state.mask_operation.unsqueeze(0) for state in states], dim=0)[index]
        mask_operation_processing = torch.cat(
            [state.mask_operation_processing.unsqueeze(0) for state in states], dim=0)[index]
        mask_operation_finished = torch.cat([state.mask_operation_finished.unsqueeze(0) for state in states], dim=0)[
            index]
        operation_adj_matrix = torch.cat([state.operation_adj_matrix.unsqueeze(0) for state in states], dim=0)[index]

        return State(operation_raw_feature=operation_raw_feature, machine_raw_feature=machine_raw_feature,
                     arc_raw_feature=arc_raw_feature,
                     mask_broken_down=mask_broken_down, mask_machine=mask_machine,
                     mask_machine_operation=mask_machine_operation,
                     mask_operation=mask_operation, mask_operation_processing=mask_operation_processing,
                     mask_operation_finished=mask_operation_finished, operation_adj_matrix=operation_adj_matrix,
                     operation_time=None, accu_matrix=None,
                     machine_broken_record=None,
                     operation_finished_time_earliest=None,
                     last_operation_per_job=None)

    def get_item(self, index):
        operation_raw_feature = self.operation_raw_feature[index]
        machine_raw_feature = self.machine_raw_feature[index]
        arc_raw_feature = self.arc_raw_feature[index]
        mask_broken_down = self.mask_broken_down[index]
        mask_machine = self.mask_machine[index]
        mask_machine_operation = self.mask_machine_operation[index]
        mask_operation = self.mask_operation[index]
        mask_operation_processing = self.mask_operation_processing[index]
        mask_operation_finished = self.mask_operation_finished[index]
        operation_adj_matrix = self.operation_adj_matrix[index]
        return State(operation_raw_feature=operation_raw_feature, machine_raw_feature=machine_raw_feature,
                     arc_raw_feature=arc_raw_feature,
                     mask_broken_down=mask_broken_down, mask_machine=mask_machine,
                     mask_machine_operation=mask_machine_operation,
                     mask_operation=mask_operation, mask_operation_processing=mask_operation_processing,
                     mask_operation_finished=mask_operation_finished, operation_adj_matrix=operation_adj_matrix,
                     operation_time=None, accu_matrix=None,
                     machine_broken_record=None,
                     operation_finished_time_earliest=None,
                     last_operation_per_job=None)

    def len(self):
        return self.operation_raw_feature.size(0)

    @staticmethod
    def cat_operation_arc(arc_feature, operation_feature):
        res = torch.cat([arc_feature, operation_feature.unsqueeze(1).expand(-1, arc_feature.size()[1], -1, -1)], dim=-1)
        return res

    @staticmethod
    def cat_machine_arc(arc_feature, machine_feature):
        res = torch.cat([arc_feature, machine_feature.unsqueeze(2).expand(-1, -1, arc_feature.size()[2], -1)], dim=-1)
        return res


class Action:
    def __init__(self, prop, runnable_cases, num_operation_max, action_index):
        super(Action, self).__init__()
        self.runnable_cases = runnable_cases.squeeze()
        if self.runnable_cases.dim() == 0:
            self.runnable_cases = self.runnable_cases.unsqueeze(0)
        self.prop = prop
        self.actions_index = action_index
        self.actions = torch.zeros(self.runnable_cases.size(0), 2, dtype=torch.long, device=prop.device)
        self.actions[:, 0] = action_index // num_operation_max
        self.actions[:, 1] = action_index % num_operation_max

    def get_action(self):
        return self.actions_index, self.prop, self.runnable_cases


class Memory:
    def __init__(self):
        super(Memory, self).__init__()
        self.state_list = []
        self.reward_list = []
        self.runnable_list = []
        self.action_index_list = []
        self.action_prop_list = []

    def add_state(self, state: State):
        state = copy.deepcopy(state).to('cpu')

        state.operation_time = None
        state.accu_matrix = None
        state.machine_broken_record = None
        state.operation_finished_time_earliest = None
        state.last_operation_per_job = None

        self.state_list.append(state)

    def add(self, reward, action_index, action_prop, runnable_cases):
        self.reward_list.append(copy.deepcopy(reward).cpu())
        self.runnable_list.append(copy.deepcopy(runnable_cases).cpu())
        self.action_index_list.append(copy.deepcopy(action_index).cpu())
        self.action_prop_list.append(copy.deepcopy(action_prop).cpu())

    def clear(self):
        self.state_list = []
        self.reward_list = []
        self.runnable_list = []
        self.action_index_list = []
        self.action_prop_list = []

    def get(self):
        state = self.state_list
        reward = torch.cat([i.unsqueeze(0) for i in self.reward_list])
        runnable = torch.cat([i.unsqueeze(0) for i in self.runnable_list])
        action_index = torch.cat([i.unsqueeze(0) for i in self.action_index_list])
        action_prop = torch.cat([i.unsqueeze(0) for i in self.action_prop_list])
        return state, reward, runnable, action_index, action_prop
