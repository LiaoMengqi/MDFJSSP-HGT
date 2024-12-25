import copy
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.base_class import State
from utils.utils import warmup_schedule, WarmupCosineAnnealingLR
from policy.hgan import HGAN



class PPO:
    def __init__(self, args, device='cpu'):

        self.policy = HGAN(args['d_machine_raw'], args['d_operation_raw'], args['d_arc_raw'], args['d_model'],
                           args['d_hidden'], args['num_layers'], args['num_head'], args['d_kv'],
                           dropout=args['dropout'], device=device)

        self.policy.to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        if args['warmup_step'] is not None:
            self.scheduler = WarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                     warmup_epochs=args['warmup_step'])
        else:
            self.scheduler = WarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                     warmup_epochs=0)
        self.MseLoss = nn.MSELoss()

        self.max_grad_norm = args['max_grad_norm']
        self.batch_size = args['batch_size']
        self.num_cases = args['num_cases']
        self.epochs = args['epochs_per_iter']
        # self.start_epoch = 0
        self.device = device

        self.gamma_m = args['gamma_m']
        self.gamma_u = args['gamma_u']
        self.clip = args['clip']
        self.plc_cft = args['policy_loss_coefficient']
        self.vf_cft = args['value_function_loss_coefficient']
        self.etp_cft = args['entropy_loss_coefficient']

        self.lambda_ = args['lambda']
        if 'accumulate_step' in args.keys():
            self.accumulate_step = args['accumulate_step']
        else:
            self.accumulate_step = None
        self.MseLoss = nn.MSELoss()

        # max min normalization
        if 'beta' in args.keys():
            self.beta = args['beta']
        else:
            self.beta = 0.99

        self.running_mean = None
        self.running_var = None

    def normalizer(self, episode_return):
        batch_mean = episode_return.mean(dim=0)
        batch_var = episode_return.var(dim=0)
        if self.running_mean is None:
            self.running_mean = batch_mean
        else:
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * batch_mean
        if self.running_var is None:
            self.running_var = batch_var
        else:
            self.running_var = self.beta * self.running_var + (1 - self.beta) * batch_var
        episode_return = (episode_return - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        return episode_return

    @torch.no_grad()
    def take_action(self, state, sample=True, training=True):
        if training:
            self.policy.train()
        else:
            self.policy.eval()
        return self.policy(state, sample=sample)

    def update(self, memory):
        self.policy.train()
        steps = len(memory.state_list)
        states, reward, runnable_cases, action_index, action_prop = memory.get()

        # calculate gains
        gain = torch.zeros(self.num_cases, 2)
        gains = []
        gamma = torch.tensor((self.gamma_m, self.gamma_u))
        for i in reversed(range(steps)):
            # x = gain[runnable_cases[i]]
            gain[runnable_cases[i]] = gain[runnable_cases[i]] * gamma + reward[i].squeeze()[runnable_cases[i]]
            gains.insert(0, copy.deepcopy(gain).unsqueeze(0))
        gains = torch.cat(gains, dim=0)
        gains[~runnable_cases] = 0
        num_cases = gains.size(1)

        # normalize gains
        for i in range(num_cases):
            mask = runnable_cases[:, i]
            x = gains[:, i][mask]
            x[0:-1, 1] = x[1:, 1]
            x[-1, 1] = 0
            gains[:, i][mask] = self.normalizer(x)

        # prepare data for training
        gains = gains[runnable_cases]
        gains = self.lambda_ * gains[:, 0] + (1 - self.lambda_) * gains[:, 1]
        action_index = action_index[runnable_cases]
        action_prop = action_prop[runnable_cases]
        states = State.package(states, runnable_cases)
        loss_list = []
        accumulate_step = 0
        self.optimizer.zero_grad()
        for epoch in range(self.epochs):
            epoch_loss = 0
            length = states.len()
            num_batches = length // self.batch_size + (length % self.batch_size != 0)
            index = torch.randperm(length)
            for batch_index in range(num_batches):
                batch_start = batch_index * self.batch_size
                batch_end = min(length, (batch_index + 1) * self.batch_size)
                batch_states = states.get_item(index[batch_start:batch_end]).to(self.device)
                batch_gains = gains[index[batch_start:batch_end]].to(self.device)
                batch_action_index = action_index[index[batch_start:batch_end]].to(self.device)
                batch_action_prop = action_prop[index[batch_start:batch_end]].to(self.device)
                prop, values, entropy = self.policy.evaluate(batch_states, batch_action_index)
                ratios = torch.exp(prop - batch_action_prop)
                advantages = batch_gains - values
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                loss = -self.plc_cft * torch.min(surr1, surr2) - self.etp_cft * entropy + self.vf_cft * self.MseLoss(
                    values, batch_gains.unsqueeze(1))
                loss = loss.mean()
                if not torch.isnan(loss):
                    loss.backward()
                    epoch_loss += loss.item()
                    accumulate_step += 1
                    # accumulate_loss += loss.item()
                    if self.accumulate_step is None or accumulate_step == self.accumulate_step:
                        accumulate_step = 0
                        self.clip_gradients()
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()
            loss_list.append(epoch_loss / num_batches)
        return sum(loss_list) / len(loss_list)

    def clip_gradients(self):
        if self.max_grad_norm is None:
            return
        parameters_to_clip = [p for p in self.policy.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=self.max_grad_norm)

    def save_policy(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.policy.state_dict(), path + '/model.pth')

    def load_policy(self, path):
        self.policy.load_state_dict(torch.load(path + '/model.pth'))
        self.policy.to(self.device)
