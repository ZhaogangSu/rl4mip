from pathlib import Path
import pyscipopt as scip
import numpy as np
import pickle
import random
import os
import time

def generate_pair(
    task, 
    problem,
    data_dir, 
    ntrain,
    nvalid,
    ntest_S,
    ntest_M,
    ntest_L,
    type_dataset
):

    data_path = os.path.join(data_dir, f'instances/{task}/lp_data/{problem}/{type_dataset}')
    sample_path = os.path.join(data_dir, f'instances/{task}/sample_data/{problem}/{type_dataset}')
    pair_path = os.path.join(data_dir, f'instances/{task}/pair_data/{problem}/{type_dataset}')

    os.makedirs(pair_path, exist_ok=True)

    st_time = time.time()

    if type_dataset == 'train':
        sample_file_path = os.path.join(sample_path, f'sample{ntrain}.pkl')
    elif type_dataset == 'valid':
        sample_file_path = os.path.join(sample_path, f'sample{nvalid}.pkl')
    # elif type_dataset == 'small_test':
    #     sample_file_path = os.path.join(sample_path, f'sample{ntest_S}.pkl')
    # elif type_dataset == 'medium_test':
    #     sample_file_path = os.path.join(sample_path, f'sample{ntest_M}.pkl')
    # elif type_dataset == 'big_test':
    #     sample_file_path = os.path.join(sample_path, f'sample{ntest_L}.pkl')
    else: 
        sample_file_path = ""

    if not os.path.exists(sample_file_path):
        print("File not found:", sample_file_path)

    else:
        with open(sample_file_path, "rb") as f:
            data = pickle.load(f)

    if type_dataset == 'train' or type_dataset == 'valid':

        solution = data[0]
        
        print(solution)

        if type_dataset == 'train':
            file_path = os.path.join(data_path, f"instance_{ntrain}.lp")
        elif type_dataset == 'valid':
            file_path = os.path.join(data_path, f"instance_{nvalid}.lp")
        else:
            file_path = ""

        if not os.path.exists(file_path):
            print("File not found:", file_path)
        
        print(file_path)

        st1_time = time.time()
        model = scip.Model()
        model.readProblem('{}'.format(file_path))
        model.hideOutput()
        ed1_time = time.time()
        print(ed1_time - st1_time)

        value_to_num = {}

        vars = model.getVars()
        constrs = model.getConss()

        ori_vars_num = len(vars)

        for indx, v in enumerate(vars):
            if indx < ori_vars_num:
                value_to_num[v.name] = indx
            else:
                value_to_num[v] = indx

        n = len(vars)
        m = len(constrs)
        print(f"n: {n}, m: {m}")

        now_solution = []
        for i in range(n):
            now_solution.append(0)

        k = [] 
        site = [] 
        value = []
        constraint = []
        constraint_type = []

        for constr in constrs:
            lhs, rhs = model.getLhs(constr), model.getRhs(constr)
            if lhs == rhs:
                constraint_type.append(3)
                constraint.append(rhs)
            elif rhs < 1e20:
                constraint_type.append(1)
                constraint.append(rhs)
            else:
                constraint_type.append(2)
                constraint.append(lhs)

            row = model.getValsLinear(constr)
            now_site = []
            now_value = []

            k.append(len(row))

            for var, coeff in row.items():
                now_site.append(value_to_num[var])
                now_value.append(coeff)
            site.append(now_site)
            value.append(now_value)

        coefficient = {}
        lower_bound = {}
        upper_bound = {}
        value_type = {}      

        variables = model.getVars()  # 保持是 SCIP Variable 对象

        obj = model.getObjective()

        for e in obj:
            vnm = e.vartuple[0].name # 目标函数
            c_o = obj[e] # 目标函数系数
            # c_o_idx = value_to_num[vnm] # 目标函数变量索引
            coefficient[vnm] = c_o
        # print(coefficient)
        for var in variables:
            var_name = var.name  # 这是字符串
            # print("var_name", var_name)
            if var.vtype() == 'BINARY':
                lower_bound[var_name] = 0
                upper_bound[var_name] = 1
            else:
                lower_bound[var_name] = 0
                upper_bound[var_name] = 1
            value_type[var_name] = var.vtype()

            # now_solution[value_to_num[var_name]] = solution[var]  # 使用变量对象作为键
            # print(var.name)
            # print(solution)
            # print(solution[var.name])
            safe_name = var.name.replace('_', '')  # 将 x_0 → x0
            now_solution[value_to_num[var_name]] = solution[safe_name]


        variable_features = []
        constraint_features = []
        edge_indices = [[], []]
        edge_features = []

        for var in variables:
            now_variable_features = []
            try:
                now_variable_features.append(coefficient[var.name])
            except KeyError:
                # 如果找不到对应的系数，可以选择加一个默认值（比如 0 或者 random）
                now_variable_features.append(0.0)  # 或者 random.random()
            now_variable_features.append(lower_bound[var.name])
            now_variable_features.append(upper_bound[var.name])
            # if value_type[var.name] == 'continuous':
            #     now_variable_features.append(0)
            # else:
            #     now_variable_features.append(1)
            now_variable_features.append(0)
            now_variable_features.append(random.random())
            variable_features.append(now_variable_features)    
        
        for i in range(m):
            now_constraint_features = []
            now_constraint_features.append(constraint[i])
            now_constraint_features.append(constraint_type[i])
            now_constraint_features.append(random.random())
            constraint_features.append(now_constraint_features)

        for i in range(m):
            for j in range(k[i]):
                edge_indices[0].append(i)
                edge_indices[1].append(site[i][j])
                edge_features.append([value[i][j]])   
        
        #图划分并打包
        partition_num = 10
        vertex_num = n + m
        edge_num = 0

        edge = []
        for i in range(vertex_num):
            edge.append([])
        for i in range(m):
            for j in range(k[i]):
                edge[site[i][j]].append(n + i)
                edge[n + i].append(site[i][j])
                edge_num += 2
        
        alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
        gamma = 1.5
        balance = 1.1
        #print(alpha)

        visit = np.zeros(vertex_num, int)
        order = []
        for i in range(vertex_num):
            if(visit[i] == 0):
                q = []
                q.append(i)
                visit[q] = 1
                now = 0
                while(now < len(q)):
                    order.append(q[now])
                    for neighbor in edge[q[now]]:
                        if(visit[neighbor] == 0):
                            q.append(neighbor)
                            visit[neighbor] = 1
                    now += 1
        

        #print(len(order))
        color = np.zeros(vertex_num, int)
        for i in range(vertex_num):
            color[i] = -1
        cluster_num = np.zeros(partition_num)

        for i in range(vertex_num):
            now_vertex = order[i]
            load_limit = balance * vertex_num / partition_num
            score = np.zeros(partition_num, float)
            for neighbor in edge[now_vertex]:
                if(color[neighbor] != -1):
                    score[color[neighbor]] += 1
            for j in range(partition_num):
                if(cluster_num[j] < load_limit):
                    score[j] -= alpha * gamma * (cluster_num[j] ** (gamma - 1))
                else:
                    score[j] = -1e9
            
            now_score = -2e9
            now_site = -1
            for j in range(partition_num):
                if(score[j] > now_score):
                    now_score = score[j]
                    now_site = j
            
            color[now_vertex] = now_site
            cluster_num[now_site] += 1

        #print(cluster_num)

        new_color = []
        for i in range(n):
            new_color.append(color[i])

        os.makedirs(pair_path, exist_ok=True)
        if type_dataset == 'train':
            with open(os.path.join(pair_path, f'pair{ntrain}.pkl'), 'wb') as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)
        else:
            with open(os.path.join(pair_path, f'pair{nvalid}.pkl'), 'wb') as f:
                pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)
        ed_time = time.time()
        print(ed_time-st_time)
    
    else:
    
        if type_dataset == 'small_test':
            file_path = os.path.join(data_path, f"instance_{ntest_S}.lp")
        elif type_dataset == 'medium_test':
            file_path = os.path.join(data_path, f"instance_{ntest_M}.lp")
        elif type_dataset == 'large_test':
            file_path = os.path.join(data_path, f"instance_{ntest_L}.lp")
        else: 
            file_path = ""

        if not os.path.exists(file_path):
            print("File not found:", file_path)
        else:
            print(file_path)

            st1_time = time.time()
            model = scip.Model()
            model.readProblem('{}'.format(file_path))
            model.hideOutput()
            ed1_time = time.time()
            print(ed1_time - st1_time)

            value_to_num = {}

            vars = model.getVars()
            constrs = model.getConss()

            ori_vars_num = len(vars)

            for indx, v in enumerate(vars):
                if indx < ori_vars_num:
                    value_to_num[v.name] = indx
                else:
                    value_to_num[v] = indx

            n = len(vars)
            m = len(constrs)
            print(f"n: {n}, m: {m}")

            now_solution = []
            for i in range(n):
                now_solution.append(0)

            k = [] 
            site = [] 
            value = []
            constraint = []
            constraint_type = []

            for constr in constrs:
                lhs, rhs = model.getLhs(constr), model.getRhs(constr)
                if lhs == rhs:
                    constraint_type.append(3)
                    constraint.append(rhs)
                elif rhs < 1e20:
                    constraint_type.append(1)
                    constraint.append(rhs)
                else:
                    constraint_type.append(2)
                    constraint.append(lhs)

                row = model.getValsLinear(constr)
                now_site = []
                now_value = []

                k.append(len(row))

                for var, coeff in row.items():
                    now_site.append(value_to_num[var])
                    now_value.append(coeff)
                site.append(now_site)
                value.append(now_value)

            coefficient = {}
            lower_bound = {}
            upper_bound = {}
            value_type = {}      

            variables = model.getVars()  # 保持是 SCIP Variable 对象

            obj = model.getObjective()

            for e in obj:
                vnm = e.vartuple[0].name # 目标函数
                c_o = obj[e] # 目标函数系数
                # c_o_idx = value_to_num[vnm] # 目标函数变量索引
                coefficient[vnm] = c_o
            # print(coefficient)
            for var in variables:
                var_name = var.name  # 这是字符串
                # print("var_name", var_name)
                if var.vtype() == 'BINARY':
                    lower_bound[var_name] = 0
                    upper_bound[var_name] = 1
                else:
                    lower_bound[var_name] = 0
                    upper_bound[var_name] = 1
                value_type[var_name] = var.vtype()

                # now_solution[value_to_num[var_name]] = solution[var]  # 使用变量对象作为键
                # print(var.name)
                # print(solution)
                # print(solution[var.name])
                # safe_name = var.name.replace('_', '')  # 将 x_0 → x0
                # now_solution[value_to_num[var_name]] = solution[safe_name]


            variable_features = []
            constraint_features = []
            edge_indices = [[], []]
            edge_features = []

            for var in variables:
                now_variable_features = []
                try:
                    now_variable_features.append(coefficient[var.name])
                except KeyError:
                    # 如果找不到对应的系数，可以选择加一个默认值（比如 0 或者 random）
                    now_variable_features.append(0.0)  # 或者 random.random()
                now_variable_features.append(lower_bound[var.name])
                now_variable_features.append(upper_bound[var.name])
                # if value_type[var.name] == 'continuous':
                #     now_variable_features.append(0)
                # else:
                #     now_variable_features.append(1)
                now_variable_features.append(0)
                now_variable_features.append(random.random())
                variable_features.append(now_variable_features)    
            
            for i in range(m):
                now_constraint_features = []
                now_constraint_features.append(constraint[i])
                now_constraint_features.append(constraint_type[i])
                now_constraint_features.append(random.random())
                constraint_features.append(now_constraint_features)

            for i in range(m):
                for j in range(k[i]):
                    edge_indices[0].append(i)
                    edge_indices[1].append(site[i][j])
                    edge_features.append([value[i][j]])   
            
            #图划分并打包
            partition_num = 10
            vertex_num = n + m
            edge_num = 0

            edge = []
            for i in range(vertex_num):
                edge.append([])
            for i in range(m):
                for j in range(k[i]):
                    edge[site[i][j]].append(n + i)
                    edge[n + i].append(site[i][j])
                    edge_num += 2
            
            alpha = (partition_num ** 0.5) * edge_num / (vertex_num ** (2 / 3))
            gamma = 1.5
            balance = 1.1
            #print(alpha)

            visit = np.zeros(vertex_num, int)
            order = []
            for i in range(vertex_num):
                if(visit[i] == 0):
                    q = []
                    q.append(i)
                    visit[q] = 1
                    now = 0
                    while(now < len(q)):
                        order.append(q[now])
                        for neighbor in edge[q[now]]:
                            if(visit[neighbor] == 0):
                                q.append(neighbor)
                                visit[neighbor] = 1
                        now += 1
            

            #print(len(order))
            color = np.zeros(vertex_num, int)
            for i in range(vertex_num):
                color[i] = -1
            cluster_num = np.zeros(partition_num)

            for i in range(vertex_num):
                now_vertex = order[i]
                load_limit = balance * vertex_num / partition_num
                score = np.zeros(partition_num, float)
                for neighbor in edge[now_vertex]:
                    if(color[neighbor] != -1):
                        score[color[neighbor]] += 1
                for j in range(partition_num):
                    if(cluster_num[j] < load_limit):
                        score[j] -= alpha * gamma * (cluster_num[j] ** (gamma - 1))
                    else:
                        score[j] = -1e9
                
                now_score = -2e9
                now_site = -1
                for j in range(partition_num):
                    if(score[j] > now_score):
                        now_score = score[j]
                        now_site = j
                
                color[now_vertex] = now_site
                cluster_num[now_site] += 1

            #print(cluster_num)

            new_color = []
            for i in range(n):
                new_color.append(color[i])

            os.makedirs(pair_path, exist_ok=True)

            if type_dataset == 'small_test':
                
                with open(os.path.join(pair_path, f'pair{ntest_S}.pkl'), 'wb') as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)

            elif type_dataset == 'medium_test':
                
                with open(os.path.join(pair_path, f'pair{ntest_M}.pkl'), 'wb') as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)

            elif type_dataset == 'big_test':

                with open(os.path.join(pair_path, f'pair{ntest_L}.pkl'), 'wb') as f:
                    pickle.dump([variable_features, constraint_features, edge_indices, edge_features, new_color, now_solution], f)

            ed_time = time.time()
            print(ed_time-st_time)