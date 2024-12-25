import argparse

from utils.utils import set_seed
from utils.case_generator import gen_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Flexible Job Scheduling')

    parser.add_argument('--data_name', type=str, default='valid_')
    # scale
    parser.add_argument('--num_machine', type=int, default=5)
    parser.add_argument('--num_job', type=int, default=15)
    parser.add_argument('--num_operation_max', type=int, default=200)
    parser.add_argument('--num_cases', type=int, default=20)
    # dynamic
    parser.add_argument('--dynamic_prop', type=float, default=0.3)
    parser.add_argument('--sigma_pres_time', type=float, default=1)
    parser.add_argument('--mu_lbd', type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed.")

    args_parsed = parser.parse_args()
    args_parsed = vars(args_parsed)
    gen_data(args_parsed)
