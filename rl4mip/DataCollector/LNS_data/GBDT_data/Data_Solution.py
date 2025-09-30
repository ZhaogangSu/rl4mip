from pathlib import Path
import pickle
import random
import time
import os
import pyscipopt as scip
import gurobipy as gp

# def get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type):
#     '''
#     Function Description:
#     Solve the problem using an optimization solver based on the provided problem instance.

#     Parameters:
#     - n: Number of decision variables in the problem instance.
#     - m: Number of constraints in the problem instance.
#     - k: k[i] represents the number of decision variables in the i-th constraint.
#     - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
#     - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
#     - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
#     - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =..
#     - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
#     - time_limit: Maximum solving time.
#     - obj_type: Whether the problem is a maximization problem or a minimization problem.

#     Return: 
#     The optimal solution of the problem, represented as a list of values for each decision variable in the optimal solution.
#     '''

#     model = gp.Model()
#     # 设置参数
#     model.setParam('OutputFlag', 0)

#     # 预先分配变量
#     x = {}
#     for i in range(n):
#         x[i] = model.addVar(vtype=gp.GRB.BINARY, lb=0, ub=1, name=f"x_{i}")

#     # 设置目标函数
#     objective = gp.quicksum(coefficient[i] * x[i] for i in range(n))
#     model.setObjective(objective, sense=gp.GRB.MAXIMIZE if obj_type == "maximize" else gp.GRB.MINIMIZE)

#     for i in range(m):
#         expr = gp.quicksum(value[i][j] * x[site[i][j]] for j in range(k[i]))
#         if constraint_type[i] == 1:
#             model.addConstr(expr <= constraint[i])
#         elif constraint_type[i] == 2:
#             model.addConstr(expr >= constraint[i])

#     model.setParam('limits/time', 600)

#     model.update()

#     model.optimize()

#     opt_sol = {}
#     opt_sol_list = []
#     mvars = model.getVars()
    
#     for var in mvars:
#         value =  var.x
#         opt_sol[var.varName] = value
#         opt_sol_list.append(value)

#     return opt_sol_list

def get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type):
    '''
    Function Description:
    Solve the problem using an optimization solver based on the provided problem instance.

    Parameters:
    - n: Number of decision variables in the problem instance.
    - m: Number of constraints in the problem instance.
    - k: k[i] represents the number of decision variables in the i-th constraint.
    - site: site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
    - value: value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    - constraint: constraint[i] represents the right-hand side value of the i-th constraint.
    - constraint_type: constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =..
    - coefficient: coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    - time_limit: Maximum solving time.
    - obj_type: Whether the problem is a maximization problem or a minimization problem.

    Return: 
    The optimal solution of the problem, represented as a list of values for each decision variable in the optimal solution.
    '''

    model = scip.Model()

    model.setParam('display/verblevel', 5)
    
    model.setParam('limits/time', time_limit)

    x = {}
    
    for i in range(n):
        x[i] = model.addVar(vtype="BINARY", lb=0, ub=1, name=f"x{i}")

    objective = scip.quicksum(coefficient[i] * x[i] for i in range(n))
    model.setObjective(objective, "maximize" if obj_type == "maximize" else "minimize")

    for i in range(m):
        expr = scip.quicksum(value[i][j] * x[site[i][j]] for j in range(k[i]))
        if constraint_type[i] == 1:
            model.addCons(expr <= constraint[i])
        elif constraint_type[i] == 2:
            model.addCons(expr >= constraint[i])

    model.optimize()

    opt_sol = {}
    opt_sol_list = []
    mvars = model.getVars()

    gap = model.getGap()

    for var in mvars:
        value =  model.getVal(var)
        opt_sol[var.name] = value
        opt_sol_list.append(value)

    return opt_sol, gap

def data_solution(
    task, 
    problem,
    data_dir, 
    ntrain,
    nvalid,
    type_dataset
):
    '''
    Function Description:
    Based on the specified parameter design, invoke the designated algorithm and solver to optimize the optimization problem in data.pickle in the current directory.

    Parameters:
    - number: Integer value indicating the number of instances to generate.
    - suboptimal: Integer value indicating the number of suboptimal solutions to generate.

    Return: 
    The optimal solution is generated and packaged as data.pickle. The function does not have a return value.
    '''
    print(f"task: {task}, problem: {problem}, data_dir: {data_dir}, type_dataset: {type_dataset}")
    pickle_files = [str(path) for path in Path(os.path.join(data_dir, f'instances/{task}/pickle_data/{problem}/{type_dataset}')).glob("*.pickle")]
    sample_path = os.path.join(data_dir, f'instances/{task}/sample_data/{problem}/{type_dataset}')
    data_path = os.path.join(data_dir, f'instances/{task}/pickle_data/{problem}/{type_dataset}')
    os.makedirs(sample_path, exist_ok=True)

    print(pickle_files)

    # for idx, instance in enumerate(pickle_files):
    if type_dataset == 'train':
        file_path = os.path.join(data_path, f"instance_{ntrain}.pickle")
    elif type_dataset == 'valid':
        file_path = os.path.join(data_path, f"instance_{nvalid}.pickle")
    else:
        file_path = ""
    
    if not os.path.exists(file_path):
        print("No input file!", file_path)
        return 
    else:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    
        # n represents the number of decision variables.
        # m represents the number of constraints.
        # k[i] represents the number of decision variables in the i-th constraint.
        # site[i][j] represents which decision variable the j-th decision variable of the i-th constraint corresponds to.
        # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
        # constraint[i] represents the right-hand side value of the i-th constraint.
        # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
        # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
        n = data[1]
        m = data[2]
        k = data[3]
        site = data[4]
        value = data[5]
        constraint = data[6]
        constraint_type = data[7]
        coefficient = data[8]
        # IS and CAT are maximization problems.
        # MVC and SC are minimization problems.
        obj_type = data[0]
        time_limit = 600  # Maximum solving time in seconds.
        optimal_solution, gap = get_best_solution(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type)
        

        # variable_features = []
        # constraint_features = []
        # edge_indices = [[], []] 
        # edge_features = []

        # for i in range(n):
        #     now_variable_features = []
        #     now_variable_features.append(coefficient[i])
        #     now_variable_features.append(0)
        #     now_variable_features.append(1)
        #     now_variable_features.append(1)
        #     now_variable_features.append(random.random())
        #     variable_features.append(now_variable_features)
        
        # for i in range(m):
        #     now_constraint_features = []
        #     now_constraint_features.append(constraint[i])
        #     now_constraint_features.append(constraint_type[i])
        #     now_constraint_features.append(random.random())
        #     constraint_features.append(now_constraint_features)
        
        # for i in range(m):
        #     for j in range(k[i]):
        #         edge_indices[0].append(i)
        #         edge_indices[1].append(site[i][j])
        #         edge_features.append([value[i][j]])
        if type_dataset == 'train':
            sample_file_path = os.path.join(sample_path, f'sample{ntrain}.pkl')
        else:
            sample_file_path = os.path.join(sample_path, f'sample{nvalid}.pkl')

        with open(sample_file_path, 'wb') as f:
                pickle.dump([optimal_solution, gap], f)
