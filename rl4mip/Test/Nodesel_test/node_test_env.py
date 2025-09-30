import sys
import os
import re
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
from functools import partial
from .env_utils import record_stats, display_stats, distribute, display_stats_2
from pathlib import Path 
import datetime
from termcolor import colored
from queue import Empty
import time


class TaskManager:
    def __init__(self, now_time, nodesels, instances, problem, nums_instances, device, normalize, verbose, default, model_path, scip_para, n_cpu, size, time_limit):

        self.now_time = now_time
        self.nodesels = nodesels
        self.instances = instances
        self.problem = problem
        self.nums_instances = nums_instances
        self.device = device
        self.normalize = normalize
        self.verbose = verbose
        self.default = default
        self.model_path = model_path
        self.scip_para = scip_para
        self.n_cpu=n_cpu
        self.size = size
        self.time_limit = time_limit

        
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.error_queue = Queue()
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
    def worker(self, worker_id):
        """工作进程函数"""
        while True:
            try:
                # 从任务队列获取任务，超时1秒
                task_data = self.task_queue.get(timeout=1)
                if task_data is None:  # 毒丸，表示没有更多任务
                    break
                
                instance_idx, instance = task_data
                # 执行原来的record_stats函数
                record_stats(
                    now_time=self.now_time,
                    nodesels=self.nodesels,
                    instances=[instance], 
                    problem=self.problem,
                    size=self.size,
                    nums_instances=self.nums_instances,
                    device=self.device,
                    normalize=self.normalize,
                    verbose=self.verbose,
                    default=self.default,
                    model_path=self.model_path,
                    scip_para=self.scip_para,
                    worker_id=worker_id,
                    time_limit=self.time_limit
                )

            except Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                break

    # def run(self):
    #     """执行所有任务"""
    #     # 将所有任务放入队列
    #     for i, instance in enumerate(self.instances):
    #         self.task_queue.put((i, instance))
        
    #     # 启动工作进程
    #     processes = []
    #     for i in range(self.n_cpu):
    #         p = Process(target=self.worker, args=(i,))
    #         p.start()
    #         processes.append(p)
        
    #     # 发送停止信号给所有工作进程
    #     for _ in range(self.n_cpu):
    #         self.task_queue.put(None)
        
    #     # 等待所有进程结束
    #     for p in processes:
    #         p.join()


    def run(self):
        """执行所有任务"""
        # 将所有任务放入队列
        for i, instance in enumerate(self.instances):
            self.task_queue.put((i, instance))
        
        # 启动工作进程
        processes = []
        try:
            for i in range(self.n_cpu):
                p = Process(target=self.worker, args=(i,))
                p.start()
                processes.append(p)
            
            # 发送停止信号给所有工作进程
            for _ in range(self.n_cpu):
                self.task_queue.put(None)
            
            # 等待所有进程结束
            for p in processes:
                p.join()
        
        finally:
            # 确保清理任何可能残留的进程
            for p in processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

class NodeselPolicyTestEnv:
    """Environment for test MIP nodes elect policies"""
    
    def __init__(self, problem, data_path, data_partition='test', seed=0):
        self.problem = problem
        self.data_path = data_path
        self.seed = seed
        self.data_partition=data_partition # test  transfer  small_transfer  medium_transfer  big_transfer

    def set_policy(self, policy, model_path='', device='cpu', n_cpu=25, normalize=True, nodesels='symb_dummy_nprimal=2', verbose=False, default=False, delete=False, scip_para=1):
        self.policy = policy
        self.device = device
        self.model_path = model_path
        self.n_cpu = n_cpu
        self.normalize = normalize
        self.verbose = verbose
        self.default = default
        self.delete = delete
        self.scip_para = scip_para
        # ['expert_dummy', 'symb_dummy_nprimal=2', 'symm_dummy_nprimal=2', 'gnn_dummy_nprimal=2', 'ranknet_dummy_nprimal=2', 'svm_dummy_nprimal=2', 'estimate_dummy']
        if 'symb' in self.model_path and 'dso' in self.model_path:
            self.nodesels = 'symb_dummy_nprimal=2'
        elif 'symm' in self.model_path and 'dso' in self.model_path:
            self.nodesels = 'symm_dummy_nprimal=2'
        elif 'gnn' in self.model_path:
            self.nodesels = 'gnn_dummy_nprimal=2'
        elif 'ranknet' in self.model_path:
            self.nodesels = 'ranknet_dummy_nprimal=2'
        elif 'svm' in self.model_path:
            self.nodesels = 'svm_dummy_nprimal=2'
        elif 'expert' in self.model_path or 'default' in self.model_path:
            self.nodesels = 'expert_dummy' # ['default_dfs_1', 'expert_dummy', 'symb_dummy_nprimal=2', 'symm_dummy_nprimal=2', 'gnn_dummy_nprimal=2', 'ranknet_dummy_nprimal=2', 'svm_dummy_nprimal=2', 'estimate_dummy']
        else:
            raise Exception("Invalid model path.")
        # TODO: 策略


    
    def test(self, size = 'small', n_instance = 4, time_limit = 3600):
        
        torch.set_default_device(self.device)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if self.delete:
            try:
                import shutil
                shutil.rmtree(os.path.join(self.data_path, '../', f'stats/{self.problem}'))
            except:
                ''
        instances = list(Path(os.path.join(self.data_path, 
                                    f"instances/{self.problem}/{size}_{self.data_partition}")).glob("*.lp"))
        # else:
        #     p = Path(self.custom_data_path)
        #     if not p.exists():
        #         print("路径不存在")
        #     elif p.is_file():            # 跟随符号链接
        #         instances = [p]
        #     elif p.is_dir():
        #         instances = sorted(p.glob("*.lp"))

        # random.shuffle(instances)

        if len(instances) < n_instance:
            print(colored('The number of test data is less than n_instance.', 'red'))
        else:
            instances = instances[:n_instance]
        
        nodesels_type = self.nodesels[0]
        str_index = nodesels_type.find('_')
        substring = nodesels_type[:str_index]

        print("Evaluation")
        print(f"  Problem:                    {self.problem}")
        print(f"  Size:                       {size}")
        print(f"  n_instance/problem:         {len(instances)}")
        print(f"  Nodeselectors evaluated:    {','.join( ['default' if self.default else '' ] + [self.nodesels])}")
        print(f"  Device for {substring} inference:   {self.device}")
        print(f"  Normalize features:         {self.normalize}")
        print("----------------")

        # processes = [Process(name=f"worker {p}", 
        #                    target=partial(record_stats,
        #                                   now_time=now_time,
        #                                   nodesels=[self.nodesels],
        #                                   instances=instances[p1:p2], 
        #                                   problem=self.problem,
        #                                   nums_instances=len(instances),
        #                                   device=torch.device(self.device),
        #                                   normalize=self.normalize,
        #                                   verbose=self.verbose,
        #                                   default=self.default,
        #                                   model_path=self.model_path,
        #                                   scip_para=self.scip_para))
        #             for p,(p1,p2) in enumerate(distribute(n_instance, self.n_cpu))]
        
        # # 'set_start_method('spawn')' 需要放在 if __name__ == "__main__": 下面
        # a = list(map(lambda p: p.start(), processes)) #run processes
        # b = list(map(lambda p: p.join(), processes)) #join processes

        # 创建TaskManager并运行
        task_manager = TaskManager(
                                    now_time=now_time,
                                    nodesels=[self.nodesels],
                                    instances=instances, 
                                    problem=self.problem,
                                    size=size,
                                    nums_instances=len(instances),
                                    device=torch.device(self.device),
                                    normalize=self.normalize,
                                    verbose=self.verbose,
                                    default=self.default,
                                    model_path=self.model_path,
                                    scip_para=self.scip_para,
                                    n_cpu=self.n_cpu,
                                    time_limit=time_limit,
        )
        task_manager.run()

        # means_nodes, res_info = display_stats(now_time, self.problem, self.data_partition, len(instances), [self.nodesels], instances, self.scip_para, default=self.default, size=size)

        # means_nodes, res_info = display_stats_2(now_time, self.problem, self.data_partition, len(instances), [self.nodesels], instances, self.scip_para, default=self.default, size=size)
        
        nodesel_substring = self.nodesels.split('_')[0]
        res_info = display_stats_2(now_time, self.problem, size, instances, self.scip_para, nodesel_substring, self.n_cpu)
        print("结果保存在：node_output.txt")
        # print(res_info)
        return ''.join(res_info)
