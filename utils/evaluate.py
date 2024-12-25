import json
import torch

from env.df_jss import DFJssEnv
from utils.case_generator import CaseGenerator
from agent.ppo import PPO


def valid(args, device, agent):
    valid_env = DFJssEnv(args, None, device=device)
    cases = json.load(open(f"./data/{args['data_name']}.json", "r"))
    cases = CaseGenerator.from_json(cases)
    valid_env.cases = cases
    state = valid_env.reset()
    terminated = False
    # print("valid ...")
    gain = None
    action_list = [[] for i in range(len(cases))]
    with torch.no_grad():
        while not terminated:
            scores, actions = agent.take_action(state, sample=False, training=False)
            cur_time = valid_env.cur_time
            state, reward, terminated, _, _ = valid_env.step(actions)
            for i in range(len(cases)):
                action_list[i].append([actions.actions[i][0].item(), actions.actions[i][1].item(), cur_time])
            if gain is None:
                gain = reward
            else:
                gain = gain + reward
    gain = torch.mean(gain.float(), dim=0)
    operation_finished_time_earliest = valid_env.state.operation_finished_time_earliest
    max_time, index = operation_finished_time_earliest.max(dim=-1)
    max_time = max_time.float().mean()
    return max_time.item(), (gain[0].item(), gain[1].item()), action_list


def eval(args, device):
    agent = PPO(args, device=device)
    agent.load_policy(args['save_path'] + args['checkpoint_id'])
    avg_makespan, avg_gain, action_list = valid(args, device, agent)
    print(f'average markspan {avg_makespan}')
    return avg_makespan, action_list
