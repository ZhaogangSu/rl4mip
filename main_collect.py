from ast import arg
import os
from torch.multiprocessing import set_start_method
import argparse
from rl4mip.DataCollector.MIPdata import MIPData

def main():
    DATA_DIR =  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    set_start_method('spawn')

    parser = argparse.ArgumentParser()

    # problem:'setcover', 'indset', 'cauctions', 'facilities'
    parser.add_argument('--problem', type=str, default='facilities', help="Problem type: setcover, indset, mvc, etc.")
    parser.add_argument('--task', type=str, default='node_selection', help="Task name, e.g., node_selection, branch, lns_CL, lns_RL")
    parser.add_argument('--ntrain', type=int, default=10, help="Number of training instances")
    parser.add_argument('--nvalid', type=int, default=10, help="Number of validation instances")
    parser.add_argument('--ntest', type=int, default=10, help="Number of test instances")
    parser.add_argument('--lns_ntrain', type=int, default=0, help="Number of lns_training instances")
    parser.add_argument('--lns_nvalid', type=int, default=0, help="Number of lns_validation instances")
    parser.add_argument('--lns_ntest_S', type=int, default=1, help="Number of lns_test_S instances")
    parser.add_argument('--lns_ntest_M', type=int, default=0, help="Number of lns_test_M instances")
    parser.add_argument('--lns_ntest_L', type=int, default=0, help="Number of lns_test_L instances")
    parser.add_argument('--dataset', type=str, default='small_test', help="type of dataset to sols")
    parser.add_argument('--neighborhood', type=int, default=3000, help="Neighborhood size for LNS")

    args_ini = parser.parse_args()    

    collector = MIPData(problem=args_ini.problem, data_dir=DATA_DIR)
    
    print(DATA_DIR)

    collector.generate_instances(
        ntrain=args_ini.ntrain,
        nvalid=args_ini.nvalid,
        ntest=args_ini.ntest,
        lns_ntrain=args_ini.lns_ntrain,
        lns_nvalid=args_ini.lns_nvalid,
        lns_ntest_S=args_ini.lns_ntest_S,
        lns_ntest_M=args_ini.lns_ntest_M,
        lns_ntest_L=args_ini.lns_ntest_L,
        task=args_ini.task
    )

    if args_ini.task == "node_selection":
        print('Collecting Node Selection Behaviours ...')
        collector.collect_node_behaviours(train_size=10, valid_size=10, test_size=0, n_cpu=10)
    
    elif args_ini.task == "branch":
        print('Collecting Full Strong Branch samples ...')
        collector.collect_branch_samples(train_size=10, valid_size=10, test_size=10, n_cpu=10)

    elif args_ini.task == "lns_GBDT" or args_ini.task == "lns_CL" or args_ini.task == "lns_RL" or args_ini.task == "lns_IL":
        collector.collect_lns_samples(
            neighborhood=args_ini.neighborhood,
            lns_ntrain=args_ini.lns_ntrain,
            lns_nvalid=args_ini.lns_nvalid,
            lns_ntest_S=args_ini.lns_ntest_S,
            lns_ntest_M=args_ini.lns_ntest_M,
            lns_ntest_L=args_ini.lns_ntest_L,
            task=args_ini.task,
            type_dataset = args_ini.dataset
        )
    else:
        raise ValueError(f"Unknown task: {args_ini.task}. Supported tasks are: node, Branch, lns_GBDT, lns_CL, lns_RL, lns_IL.")
    
if __name__ == '__main__':
    main()