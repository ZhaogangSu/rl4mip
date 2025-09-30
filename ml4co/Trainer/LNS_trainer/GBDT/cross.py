import numpy as np
from ml4co.Trainer.LNS_trainer.GBDT.gurobi_solver import Gurobi_solver

def cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, solA, blockA, solB, blockB, set_time, lower_bound, upper_bound, value_type):
    crossX = np.zeros(n)
    
    for i in range(n):
        if(blockA[i] == 1):
            crossX[i] = solA[i]
        else:
            crossX[i] = solB[i]
    
    color = np.zeros(n)
    add_num = 0
    for j in range(m):
        constr = 0
        flag = 0
        for l in range(k[j]):
            if(color[site[j][l]] == 1):
                flag = 1
            else:
                constr += crossX[site[j][l]] * value[j][l]

        if(flag == 0):
            if(constraint_type[j] == 1):
                if(constr > constraint[j]):
                    for l in range(k[j]):
                        if(color[site[j][l]] == 0):
                            color[site[j][l]] = 1
                            add_num += 1
            else:
                if(constr < constraint[j]):
                    for l in range(k[j]):
                        if(color[site[j][l]] == 0):
                            color[site[j][l]] = 1
                            add_num += 1
    if(add_num / n <= rate):
        newcrossX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, set_time, obj_type, lower_bound, upper_bound, value_type, crossX, color)
        return newcrossX, newVal
    else:
        return -1, -1