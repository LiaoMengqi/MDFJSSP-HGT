import argparse

from utils.utils import set_device, set_seed
from utils.case_generator import gen_data
from utils.train import train


def main(args_parsed):
    device = set_device(args_parsed['gpu'])
    if args_parsed['seed'] is not None:
        set_seed(args_parsed['seed'])
        train(args_parsed, device)
    else:
        train(args_parsed, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Flexible Job Scheduling')
    # main
    parser.add_argument('--data_name', type=str, default="valid")
    parser.add_argument('--save_path', type=str, default="./checkpoint/")
    parser.add_argument('--checkpoint_id', type=str, default="default")
    # environment
    parser.add_argument('--num_machine', type=int, default=10)
    parser.add_argument('--num_job', type=int, default=20)
    parser.add_argument('--num_operation_max', type=int, default=200)
    parser.add_argument('--num_cases', type=int, default=10)
    parser.add_argument('--dynamic_prop', type=float, default=0.3)
    parser.add_argument('--sigma_pres_time', type=float, default=1)
    parser.add_argument('--mu_lbd', type=float, default=0.01)
    # hgt policy
    parser.add_argument('--d_operation_raw', type=int, default=7)
    parser.add_argument('--d_machine_raw', type=int, default=4)
    parser.add_argument('--d_arc_raw', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_hidden', type=int, default=256)
    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--d_kv', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)
    # ppo
    parser.add_argument('--gamma_m', type=float, default=0.925)
    parser.add_argument('--gamma_u', type=float, default=0.925)
    parser.add_argument('--lambda', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--policy_loss_coefficient', type=float, default=1)
    parser.add_argument('--value_function_loss_coefficient', type=float, default=1)
    parser.add_argument('--entropy_loss_coefficient', type=float, default=0.01)
    # train args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--case_regen_iter', type=int, default=5)
    parser.add_argument('--epochs_per_iter', type=int, default=3)
    parser.add_argument('--early_stop', type=int, default=None)
    parser.add_argument('--warmup_step', type=int, default=None)
    parser.add_argument('--accumulate_step', type=int, default=None)
    # general setting
    parser.add_argument("--gpu", type=int, default=-2,
                        help="Use the GPU with the lowest memory footprint by default. "
                             "Specify a GPU by setting this parameter to a GPU id which equal to or greater than 0."
                             "Set this parameter to _1 to use the CPU."
                        )
    parser.add_argument("--seed", type=int, default=None, help="random seed.")
    args_parsed = parser.parse_args()
    args_parsed = vars(args_parsed)
    main(args_parsed)