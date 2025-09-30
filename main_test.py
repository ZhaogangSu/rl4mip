import os
import torch
import sys
import argparse
from torch.multiprocessing import set_start_method
from ml4co.Test.MIPtest import MIPTest


def main():
    DATA_DIR =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    MODEL_DIR =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="A script to handle size and instances parameters")

    parser.add_argument('task', type=str, nargs='?', default='branch', help='task type (default: branch)')
    parser.add_argument('problem', type=str, nargs='?', default='setcover', help='Instance type (default: setcover)')
    parser.add_argument('method', type=str, nargs='?', default='gnn', help='Testing methods (default: lns_CL)')
    parser.add_argument('size', type=str, nargs='?', default='small', help='Problem size (default: small)')

    args = parser.parse_args()

    evaluation = MIPTest(task=args.task, problem=args.problem, data_dir=DATA_DIR, model_dir=MODEL_DIR)
    results = evaluation.test(method=args.method, device=DEVICE, size=args.size, n_instance=1, time_limit=60, n_cpu=1)
    print(results)

if __name__ == '__main__':
    set_start_method('spawn')
    main()