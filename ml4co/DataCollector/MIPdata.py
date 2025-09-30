import os
import multiprocessing
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MIPData:
    
    def __init__(self, problem, data_dir):

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.problem = problem
    def generate_instances(self, ntrain=10, nvalid=10, ntest=10, lns_ntrain=1, lns_nvalid=1, lns_ntest_S=1, lns_ntest_M=1, lns_ntest_L=1, task='lns_GBDT'):
        
        if task == "branch" or task == "node_selection":
            if self.problem in ['setcover', 'indset', 'cauctions', 'facilities', 'maxcut']:
                from Branch_data.InstancesGenerator import InstancesGen
                InstancesGen(
                    problem=self.problem, 
                    ntrain=ntrain, 
                    nvalid=nvalid, 
                    ntest=ntest,
                    seed=0,
                    datapath=self.data_dir
                    )
                
            
            if self.problem in ['GISP', 'FCMCNF', 'WPMS']:
                from Nodesel_data.problem_generation import problem_generator
                n_cpu = self._get_optimal_cpu_count()
                problem_generator(
                    problem=self.problem, 
                    ntrain=ntrain, 
                    nvalid=nvalid, 
                    ntest=ntest, 
                    n_cpu=n_cpu,
                    datapath=self.data_dir
                    )
        elif task == 'lns_CL':
            if self.problem in ['setcover', 'indset', 'cauctions', 'mvc']:
                from LNS_data.CL_data.InstanceGenerator import InstancesGen_CL
                InstancesGen_CL(
                    task=task,
                    problem=self.problem, 
                    ntrain=lns_ntrain, 
                    nvalid=lns_nvalid,
                    ntest_S=lns_ntest_S,
                    ntest_L=lns_ntest_L,
                    seed=0,
                    datapath=self.data_dir
                )

        elif task == 'lns_GBDT':
            if self.problem in ['setcover', 'indset', 'cauctions', 'mvc']:
                from LNS_data.GBDT_data.InstanceGenerator import InstancesGen_GBDT
                InstancesGen_GBDT(
                    task=task,
                    problem=self.problem, 
                    ntrain=lns_ntrain, 
                    nvalid=lns_nvalid,
                    ntest_S=lns_ntest_S,
                    ntest_M=lns_ntest_M,
                    ntest_L=lns_ntest_L,
                    seed=0,
                    datapath=self.data_dir
                )
        elif task == 'lns_IL':
            if self.problem in ['setcover', 'indset', 'cauctions', 'mvc']:
                from LNS_data.IL_data.InstanceGenerator import InstancesGen_IL
                InstancesGen_IL(
                    task=task,
                    problem=self.problem, 
                    ntrain=lns_ntrain, 
                    nvalid=lns_nvalid,
                    ntest_S=lns_ntest_S,
                    ntest_L=lns_ntest_L,
                    seed=0,
                    datapath=self.data_dir
                )
        else:
            if self.problem in ['setcover', 'indset', 'cauctions', 'maxcut']:
                from LNS_data.RL_data.InstancesGenerator import InstancesGen_RL
                InstancesGen_RL(
                    task=task,
                    problem=self.problem, 
                    ntrain=lns_ntrain, 
                    nvalid=lns_nvalid,
                    ntest_S=lns_ntest_S,
                    ntest_M=lns_ntest_M,
                    ntest_L=lns_ntest_L,
                    seed=0,
                    datapath=self.data_dir
                )

    def collect_branch_samples(self, train_size=100000, valid_size=20000, test_size=100, n_cpu=None):

        from Branch_data.FSBdataCollector import FSBCollector
        
        FSBCollector(problem=self.problem, train_size=train_size, valid_size=valid_size, test_size=test_size, 
                        datapath=self.data_dir, 
                        njobs=n_cpu or self._get_optimal_cpu_count())
    
    def collect_node_behaviours(self, train_size=10, valid_size=10, test_size=10, n_cpu=None):

        from Nodesel_data.behaviour_generation import behaviour_generator
        from Nodesel_data.problem_generation import sol_generator

        if self.problem in ['setcover', 'indset', 'cauctions', 'facilities', 'maxcut']:
            sol_generator(problem=self.problem, ntrain=train_size, nvalid=valid_size, ntest=test_size,
                                n_cpu=n_cpu or self._get_optimal_cpu_count(), 
                                datapath=self.data_dir)
        
        behaviour_generator(problem=self.problem, ntrain=train_size, nvalid=valid_size, ntest=test_size,
                                n_cpu=n_cpu or self._get_optimal_cpu_count(), 
                                datapath=self.data_dir)
    
    def collect_lns_samples(self, neighborhood, ntrain=1, nvalid=1, lns_ntrain=1, lns_nvalid=1, lns_ntest_S=1, lns_ntest_M=1, lns_ntest_L=1, task='lns_RL', type_dataset = 'train'):
        
        if task == 'lns_CL':
            if self.problem in ['setcover', 'indset', 'cauctions', 'mvc']:
                if type_dataset in ['train', 'valid']:
                    from LNS_data.CL_data.IMCollector import IMCollector
                    IMCollector(neighborhood, task, self.problem, ntrain, nvalid, self.data_dir)
            
        if task == 'lns_IL':
            if self.problem in ['setcover', 'indset', 'cauctions', 'mvc']:
                if type_dataset in ['train', 'valid']:
                    from LNS_data.IL_data.IMCollector import IMCollector
                    IMCollector(neighborhood, task, self.problem, ntrain, nvalid, self.data_dir)
                    from LNS_data.IL_data.pre_Collector_utils import collect_sols
                    collect_sols(task, self.problem, self.data_dir, type_dataset)

        if task == 'lns_GBDT':
            from LNS_data.GBDT_data.Data_Solution import data_solution
            from LNS_data.GBDT_data.pairData_Collector import generate_pair
            data_solution(task, self.problem, self.data_dir, ntrain=lns_ntrain, nvalid=lns_nvalid, type_dataset=type_dataset)
            generate_pair(task, self.problem, self.data_dir, ntrain=lns_ntrain, nvalid=lns_nvalid, ntest_S = lns_ntest_S, ntest_M = lns_ntest_M, ntest_L = lns_ntest_L, type_dataset=type_dataset)

    def _get_optimal_cpu_count(self):
        return min(max(1, multiprocessing.cpu_count() - 1), 50)



        

    

