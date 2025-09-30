# from rl4mip.Test.Nodesel_test.node_test_env import NodeselPolicyTestEnv
# from rl4mip.Test.Branch_test.BranchTestEnv import BranchPolicyTestEnv
# from rl4mip.Test.LNS_test.LNSTtestEnv import LNSPolicyTestEnv, NDPolicyTestEnv, optimize
# import os

# class MIPTest:
#     def __init__(self, task, problem, data_dir=None, model_dir=None, scip_para=5):
#         self.task = task
#         self.model_dir = model_dir
#         self.problem = problem
#         self.scip_para = scip_para
#         if self.task == 'node_selection':
#             self.test_env = NodeselPolicyTestEnv(problem=problem, data_path=data_dir)
#         elif self.task == 'branch':
#             self.test_env = BranchPolicyTestEnv(problem=problem, data_path=data_dir)
#         elif self.task == 'LNS':
#             self.test_env = LNSPolicyTestEnv(problem, data_path=data_dir, path_save_results=path_save_results)
#             self.test_pre = NDPolicyTestEnv(problem)
#         else:
#             raise NotImplementedError

#     def test(self, method, model_path=None, device='cpu', size='small', n_instance=20, time_limit=300, n_cpu=10):
#         if self.task == 'node_selection':
#             if method != 'symb' and method != 'symm':
#                 model_path = model_path or os.path.join(self.model_dir, 'node_selection', 'policy_'+self.problem+"_"+method+".pkl")
#             else:
#                 model_path = model_path or os.path.join(self.model_dir, 'node_selection', 'policy_dso_'+method+".json")
#             self.test_env.set_policy(policy=method, model_path=model_path, default=False, delete=False, device=device, n_cpu=n_cpu, scip_para=self.scip_para)
#             results = self.test_env.test(size=size, n_instance=n_instance)
#             return results

#         elif self.task == 'branch':
#             if method == 'gnn':
#                 m_path = model_path or os.path.join(self.model_dir, 'gnn_policy', f'{self.problem}.pkl')
#                 model_name = None
#             elif method == 'symb':
#                 m_path = model_path or os.path.join(self.model_dir, 'symb_policy', f'{self.problem}.txt')
#                 model_name = None
#             elif method == 'graph':
#                 m_path = model_path or os.path.join(self.model_dir, 'graph_policy', f'{self.problem}.txt')
#                 model_name = None
#             elif method == 'hybrid':
#                 # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'gnn_params.pkl')
#                 # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.01_params.pkl')
#                 # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.001_params.pkl')
#                 # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.0001_params.pkl')

#                 m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_MHE_0.001_params.pkl')
#                 model_name = 'film'

#             self.test_env.set_policy(policy=method, device=device, model_path=m_path, model_name=model_name, scip_para=self.scip_para, n_cpu=n_cpu)
#             results = self.test_env.test(size=size, n_instance=n_instance, time_limit=time_limit)
#             return results
        
#         elif self.task == 'LNS':
#             if method == 'lns_CL':
#                 model_path = model_path or os.path.join(self.model_dir, method, self.problem, f'model_{self.problem}_feat2_nt_xent', f'neural_LNS_{self.problem}_feat2_gat.pt_best')
#                 self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path, pre_model_path = None)
#                 results = self.test_env.test(size=size, n_instance = n_instance, multi_instance = multi_instance, time_limit=time_limit)
#                 return results
#             elif method == 'lns_IL':
#                 pre_model_path = os.path.join(self.model_dir, method, self.problem, f'Pre_model_{self.problem}', 'model_best.pth')
#                 model_path = os.path.join(self.model_dir, method, self.problem, f'model_{self.problem}_feat2_bce', f'neural_LNS_{self.problem}_feat2_gat.pt')
#                 self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path, pre_model_path = pre_model_path)
#                 results = self.test_env.test(size=size, n_instance = n_instance, time_limit=time_limit)
#             elif method == "lns_GBDT":
#                 model_path_GNN = model_path or os.path.join(self.model_dir, method, self.problem, f'GNN_{self.problem}.pkl')
#                 model_path_GBDT = model_path or os.path.join(self.model_dir, method, self.problem, f'GBDT_{self.problem}.pkl')
#                 pass
#             elif method == "lns_RL":
#                 pass
#         else:
#             raise NotImplementedError

from rl4mip.Test.Nodesel_test.node_test_env import NodeselPolicyTestEnv
from rl4mip.Test.Branch_test.BranchTestEnv import BranchPolicyTestEnv
from rl4mip.Test.LNS_test.LNSTtestEnv import LNSPolicyTestEnv, NDPolicyTestEnv, optimize
import os

class MIPTest:
    def __init__(self, task, problem, data_dir, model_dir, path_save_results='./test_LNS', scip_para=5):
        self.task = task
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.problem = problem
        self.path_save_results = path_save_results

        self.scip_para = scip_para
        if self.task == 'node_selection':
            self.test_env = NodeselPolicyTestEnv(problem=problem, data_path=data_dir)
        elif self.task == 'branch':
            self.test_env = BranchPolicyTestEnv(problem=problem, data_path=data_dir)
        elif self.task == 'LNS':
            self.test_env = LNSPolicyTestEnv(problem, data_path=data_dir, path_save_results=path_save_results)
            self.test_pre = NDPolicyTestEnv(problem)
        else:
            raise NotImplementedError

    def test(self, method, model_path=None, device='cpu', size='small', n_instance=20, time_limit=3600, n_cpu=10):
        if self.task == 'node_selection':
            if method != 'symb' and method != 'symm':
                model_path = model_path or os.path.join(self.model_dir, 'node_selection', 'policy_'+self.problem+"_"+method+".pkl")
            else:
                model_path = model_path or os.path.join(self.model_dir, 'node_selection', 'policy_dso_'+method+".json")
            self.test_env.set_policy(policy=method, model_path=model_path, default=False, delete=False, device=device, n_cpu=n_cpu, scip_para=self.scip_para)
            results = self.test_env.test(size=size, n_instance=n_instance, time_limit=time_limit)
            return results

        elif self.task == 'branch':
            if method == 'gnn':
                m_path = model_path or os.path.join(self.model_dir, 'gnn_policy', f'{self.problem}.pkl')
                model_name = None
            elif method == 'symb':
                m_path = model_path or os.path.join(self.model_dir, 'symb_policy', f'{self.problem}.txt')
                model_name = None
            elif method == 'graph':
                m_path = model_path or os.path.join(self.model_dir, 'graph_policy', f'{self.problem}.txt')
                model_name = None
            elif method == 'hybrid':
                # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'gnn_params.pkl')
                # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.01_params.pkl')
                # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.001_params.pkl')
                # m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.0001_params.pkl')
                if self.problem == 'setcover':
                    m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_ED_0.0001_params.pkl')
                else:
                    m_path = model_path or os.path.join(self.model_dir, 'hybrid_policy', self.problem, 'film_distilled_MHE_0.0001_params.pkl')

                model_name = 'film'

            self.test_env.set_policy(policy=method, device=device, model_path=m_path, model_name=model_name, scip_para=self.scip_para, n_cpu=n_cpu)
            results = self.test_env.test(size=size, n_instance=n_instance, time_limit=time_limit)
            return results
        
        elif self.task == 'LNS':
            if method == 'lns_CL':
                model_path = model_path or os.path.join(self.model_dir, method, self.problem, f'model_{self.problem}_feat2_nt_xent', f'neural_LNS_{self.problem}_feat2_gat.pt_best')
                self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path, pre_model_path = None)
                results = self.test_env.test(size=size, n_instance = n_instance, multi_instance = n_cpu, time_limit=time_limit)
            
            elif method == 'lns_IL':
                pre_model_path = os.path.join(self.model_dir, method, self.problem, f'Pre_model_{self.problem}', 'model_best.pth')
                model_path = os.path.join(self.model_dir, method, self.problem, f'model_{self.problem}_feat2_bce', f'neural_LNS_{self.problem}_feat2_gat.pt_best')
                self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path, pre_model_path = pre_model_path)
                results = self.test_env.test(size=size, n_instance = n_instance, multi_instance = n_cpu, time_limit=time_limit)
            
            elif method == "lns_GBDT":
                model_path_GNN = model_path or os.path.join(self.model_dir, method, self.problem, f'GNN_{self.problem}.pkl')
                model_path_GBDT = model_path or os.path.join(self.model_dir, method, self.problem, f'GBDT_{self.problem}.pkl')
                self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path_GNN, pre_model_path = model_path_GBDT)
                results = self.test_env.test(size=size, n_instance = n_instance, multi_instance = n_cpu, time_limit=time_limit)
                
            elif method == "lns_RL":
                model_path = model_path or os.path.join(self.model_dir, method, self.problem, "gnn", f"{self.problem}.pt")
                self.test_env.set_policy(task=self.task, policy=method, device=device, model_path=model_path, pre_model_path=None)
                results = self.test_env.test(size=size, n_instance=n_instance, multi_instance=n_cpu, time_limit=time_limit)
            return results
        
        else:
            raise NotImplementedError
        
        