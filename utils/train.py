import json

import torch
from utils.evaluate import valid
from agent.ppo import PPO
from env.df_jss import DFJssEnv
import utils.base_class

from tqdm import tqdm


def train(args, device):
    env = DFJssEnv(args, None, device=device)
    agent = PPO(args, device=device)
    # rand_t = valid(args, device, agent)
    best = 10000000
    step_left = 0
    makespan_list = []
    gain_list = []
    loss_list = []
    if args['early_stop'] is not None:
        step_left = args['early_stop']

    pbar = tqdm(total=args['iterations'])
    for iteration in range(args['iterations']):
        if iteration % args['case_regen_iter'] == 0:
            # print("\n--------re-generate train cases--------\n")
            state = env.reset(keep_cases=False)
        else:
            state = env.reset(keep_cases=True)
        terminated = False
        memory = utils.base_class.Memory()
        # print(f"iteration {iteration + 1}: collecting trajectory ...")
        with torch.no_grad():
            while not terminated:
                memory.add_state(state)
                scores, actions = agent.take_action(state)
                action_index, action_prop, runnable_cases = actions.get_action()
                state, reward, terminated, _, _ = env.step(actions)
                memory.add(reward, action_index, action_prop, runnable_cases)
        # print("update policy ...")
        loss = agent.update(memory)
        memory.clear()
        v_t, gain, _ = valid(args, device, agent)
        loss_list.append(loss)
        makespan_list.append(v_t)
        gain_list.append(gain)
        # early stop
        if v_t < best:
            best = v_t
            agent.save_policy(args['save_path'] + args['checkpoint_id'] + '/')
            if args['early_stop'] is not None:
                step_left = args['early_stop']
        else:
            step_left -= 1
            if args['early_stop'] is not None and step_left <= 0:
                break

        pbar.update(1)
        pbar.set_postfix(loss=loss, makespan=v_t, best=best)
        info = {'loss': loss_list, 'makespan': makespan_list, 'gain': gain_list}
        json.dump(info, open(args['save_path'] + args['checkpoint_id'] + '/' + 'info.json', 'w'))
    pbar.close()
    print('finished')
