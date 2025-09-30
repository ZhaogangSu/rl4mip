import time 
import pyscipopt as scip

def Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, time_limit, obj_type, lower_bound, upper_bound, value_type, now_sol, now_col):
    '''
    函数说明：
    根据传入的问题实例，使用Gurobi求解器进行求解。

    参数说明：
    - n: 问题实例的决策变量数量。
    - m: 问题实例的约束数量。
    - k: k[i]表示第i条约束的决策变量数量。
    - site: site[i][j]表示第i个约束的第j个决策变量是哪个决策变量。
    - value: value[i][j]表示第i个约束的第j个决策变量的系数。
    - constraint: constraint[i]表示第i个约束右侧的数。
    - constraint_type: constraint_type[i]表示第i个约束的类型，1表示<=，2表示>=
    - coefficient: coefficient[i]表示第i个决策变量在目标函数中的系数。
    - time_limit: 最大求解时间。
    - obj_type: 问题是最大化问题还是最小化问题。
    '''

    #获得起始时间
    begin_time = time.time()

    #定义求解模型
    model = scip.Model()
    model.setParam('display/verblevel', 0)

    #设定变量映射
    site_to_new = {}
    new_to_site = {}
    new_num = 0
    x = []
    for i in range(n):
        if(now_col[i] == 1):
            site_to_new[i] = new_num
            new_to_site[new_num] = i
            new_num += 1
            if(value_type[i] == 'Binary'):
                x.append(model.addVar(vtype="Binary", lb=lower_bound[i], ub=upper_bound[i], name=f"x_{i}"))
            elif(value_type[i] == 'Continuous'):
                x.append(model.addVar(vtype="Continuous", lb=lower_bound[i], ub=upper_bound[i], name=f"x_{i}"))
            else:
                x.append(model.addVar(vtype="INTEGER", lb=lower_bound[i], ub=upper_bound[i], name=f"x_{i}"))
    
    #设定目标函数和优化目标（最大化/最小化）
    objective = 0

    for i in range(n):
        if(now_col[i] == 1):
            objective += x[site_to_new[i]] * coefficient[i]
        else:
            objective += now_sol[i] * coefficient[i]

    model.setObjective(objective, "maximize" if obj_type == "maximize" else "minimize")

    #添加m条约束

    for i in range(m):
        constr = 0
        flag = 0
        for j in range(k[i]):
            if(now_col[site[i][j]] == 1):
                constr += x[site_to_new[site[i][j]]] * value[i][j]
                flag = 1
            else:
                constr += now_sol[site[i][j]] * value[i][j]

        if(flag == 1):
            if(constraint_type[i] == 1):
                model.addCons(constr <= constraint[i])
            elif(constraint_type[i] == 2):
                model.addCons(constr >= constraint[i])
            else:
                model.addCons(constr == constraint[i])
        else:
            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
            else:
                if(constr < constraint[i]):
                    print("QwQ")
                    print(constr,  constraint[i])
                    print(now_col)
    #设定最大求解时间
    model.setParam('limits/time', max(time_limit - (time.time() - begin_time), 0))
    # model.setParam('TimeLimit', max(time_limit - (time.time() - begin_time), 0))
    #优化求解
    model.optimize()
    try:
        new_sol = []
        for i in range(n):
            if(now_col[i] == 0):
                new_sol.append(now_sol[i])
            else:
                if(value_type[i] == 'CONTINUOUS'):
                    new_sol.append(model.getVal(x[site_to_new[i]]))
                else:
                    new_sol.append(model.getVal(x[site_to_new[i]]))
            
        return new_sol, model.getObjVal() # 解和目标值
    except:
        return [], -1