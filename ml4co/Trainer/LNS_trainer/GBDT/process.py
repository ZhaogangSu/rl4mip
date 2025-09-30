import time
import torch
import numpy as np
import random
import cloudpickle as pickle
from functools import cmp_to_key
from model.graphcnn import GNNPolicy
from model.gbdt_regressor import GradientBoostingRegressor
from Envs.generate_blocks import cmp, pair
from Envs.gurobi_solver import Gurobi_solver
from Envs.cross import cross
from Envs.generate_blocks import random_generate_blocks, cross_generate_blocks

def optimize(pair_path: str,
             pickle_path: str,
             graph_model_path: str,
             gbdt_model_path: str,
             fix : float,
             set_time : int,
             rate : float,
             device: str):
    
    begin_time = time.time()

    with open(pickle_path, "rb") as f:
        matrix = pickle.load(f)

    with open(pair_path, "rb") as f:
        pairs = pickle.load(f)

    policy = GNNPolicy().to(device)
    # policy.load_state_dict(torch.load(model_path, policy.state_dict()))
    policy.load_state_dict(torch.load(graph_model_path, map_location=device))
    logits = policy(
        torch.FloatTensor(pairs[1]).to(device),
        torch.LongTensor(pairs[2]).to(device),
        torch.FloatTensor(pairs[3]).to(device),
        torch.FloatTensor(pairs[0]).to(device),
    )
 
    data = logits.tolist()

    obj_type = matrix[0]
    n = matrix[1]
    m = matrix[2]
    k = matrix[3]
    site = matrix[4]
    value = matrix[5]
    constraint = matrix[6]
    constraint_type = matrix[7]
    coefficient = matrix[8]
    lower_bound = matrix[9]
    upper_bound = matrix[10]
    value_type = matrix[11]

    # with open(gbdt_model_path, "rb") as f:
    with open(gbdt_model_path, "rb") as f:
        GBDT = pickle.load(f)[0]

    b_time = time.time()
    predict = GBDT.predict(np.array(data)) # 预测初始解
    loss =  GBDT.calc(np.array(data))
    e_time = time.time()
    print("predict time: ", e_time - b_time)
    values = []
    for i in range(n):
        values.append(pair())
        values[i].site= i
        values[i].loss = loss[i]
    
    random.shuffle(values)
    values.sort(key = cmp_to_key(cmp))  

    set_rate = 1
    
    for turn in range(10):
        
        obj = (int)(n * (1 - set_rate * rate))

        solution = []
        color = []
        for i in range(n):
            solution.append(0) # 解
            color.append(0) # 颜色

        for i in range(n):
            now_site = values[i].site
            if(i < obj):
                if(predict[now_site] < 0.5):
                    solution[now_site] = 0
                else:
                    solution[now_site] = 1
            else:
                color[now_site] = 1

        for i in range(m):
            constr = 0
            flag = 0
            for j in range(k[i]):
                if(color[site[i][j]] == 1):
                    flag = 1
                else:
                    constr += solution[site[i][j]] * value[i][j]

            if(constraint_type[i] == 1):
                if(constr > constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            constr -= solution[site[i][j]] * value[i][j]
            else:
                if(constr + flag < constraint[i]):
                    for j in range(k[i]):
                        if(color[site[i][j]] == 0):
                            color[site[i][j]] = 1
                            obj -= 1
                            break
        print(obj / n)
        if(obj / n + rate >= 1):
            break
        else:
            set_rate -= 0.1

    #初始解
    ansTime = []
    ansVal = []
    nowX, nowVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, (0.5 * set_time), obj_type, lower_bound, upper_bound, value_type, solution, color) 
    #print("nowX", nowX)
    #print("nowVal", nowVal)
    ansTime.append(time.time() - begin_time)
    ansVal.append(nowVal)

    
    random_flag = 0
    while(time.time() - begin_time < set_time):

        if(random_flag == 1):
            turnX = []
            for i in range(n):
                turnX.append(0)
            if(obj_type == 'maximize'):
                turnVal = 0
            else:
                turnVal = 1e9
            block_list, score, _ = random_generate_blocks(n, m, k, site, values, loss, fix, rate, predict, nowX)
            neibor_num = len(block_list)
            for turn in range(int(1 / rate)):
                i = 0
                now_loss = 0
                for j in range(3):
                    now_site = random.randint(0, neibor_num - 1)
                    if(score[now_site] > now_loss):
                        now_loss = score[now_site]
                        i = now_site
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                if(obj_type == 'maximize'):
                    if(newVal > turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
                else:
                    if(newVal < turnVal):
                        turnVal = newVal
                        for j in range(n):
                            turnX[j] = newX[j]
            if(obj_type == 'maximize'):
                if(turnVal == 0):
                    continue
            else:
                if(turnVal == 1e9):
                    continue

            for i in range(n):
                nowX[i] = turnX[i]
            nowVal = turnVal
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
        else:
            turnX = []
            turnVal = []
            block_list, _, _ = cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data)
            for i in range(4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = Gurobi_solver(n, m, k, site, value, constraint, constraint_type, coefficient, min(max_time, 0.2 * set_time), obj_type, lower_bound, upper_bound, value_type, nowX, block_list[i])
                if(newVal == -1):
                    continue
                turnX.append(newX)
                turnVal.append(newVal)
            
            #cross
            if(len(turnX) == 4):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[0], block_list[0], turnX[1], block_list[1], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)

                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[2], block_list[2], turnX[3], block_list[3],  min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newVal != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            if(len(turnX) == 6):
                max_time = set_time - (time.time() - begin_time)
                if(max_time <= 0):
                    break

                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[0][i] == 1 or block_list[1][i] == 1):
                        block_list[4][i] = 1
                block_list.append(np.zeros(n, int))
                for i in range(n):
                    if(block_list[2][i] == 1 or block_list[3][i] == 1):
                        block_list[5][i] = 1
                
                newX, newVal = cross(n, m, k, site, value, constraint, constraint_type, coefficient, obj_type, rate, turnX[4], block_list[4], turnX[5], block_list[5], min(max_time, 0.2 * set_time), lower_bound, upper_bound, value_type)
                if(newX != -1):
                    turnX.append(newX)
                    turnVal.append(newVal)
            
            for i in range(len(turnVal)):
                if(obj_type == 'maximize'):
                    if(turnVal[i] > nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
                else:
                    if(turnVal[i] < nowVal):
                        nowVal = turnVal[i]
                        for j in range(n):
                            nowX[j] = turnX[i][j]
            
            ansTime.append(time.time() - begin_time)
            ansVal.append(nowVal)
            # print(rate, "IS")
            print("time:", time.time() - begin_time, "nowVal:", nowVal)
    print(ansTime)
    print(ansVal)
    return ansTime, ansVal