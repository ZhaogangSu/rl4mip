import numpy as np
import random
from functools import cmp_to_key

class pair: 
    def __init__(self): 
        self.site = 0
        self.loss = 0

def cmp(a, b): # 定义cmp
    if a.loss < b.loss: 
        return -1 # 不调换
    else:
        return 1 # 调换

def cmp2(a, b): # 定义cmp
    if a.loss > b.loss: 
        return -1 # 不调换
    else:
        return 1 # 调换

def random_generate_blocks(n, m, k, site, values, loss, fix, rate, predict, nowX):
    parts = []
    score = []
    number = []
    parts.append(np.zeros(n))
    score.append(0)
    number.append(0)

    fix_num = (int)(fix * n)
    
    fix_color = np.zeros(n)
    for i in range(fix_num):
        fix_color[values[i].site] = 1
    
    now_site = 0
    for i in range(m):
        new_num = number[now_site]
        for j in range(k[i]):
            if(parts[now_site][site[i][j]] == 0):
                new_num += 1
        if(new_num > (int)(rate * n)):
            now_site += 1
            parts.append(np.zeros(n))
            score.append(0)
            number.append(0)
        
        for j in range(k[i]):
            if(parts[now_site][site[i][j]] == 0):
                parts[now_site][site[i][j]] = 1
                score[now_site] += loss[site[i][j]] * abs(nowX[site[i][j]] - predict[site[i][j]])
                number[now_site] += 1
    
    return(parts, score, number)

def cross_generate_blocks(n, loss, rate, predict, nowX, GBDT, data):
    parts = []
    score = []
    number = []
    for i in range(4):
        parts.append(np.zeros(n))
        score.append(0)
        number.append(0)
    
    pairs = []
    for i in range(n):
        pairs.append(pair())
        pairs[i].site= i
        pairs[i].loss = loss[i] * abs(nowX[i] - predict[i])
    pairs.sort(key = cmp_to_key(cmp2))  
    
    now_tree = random.randint(0, 5)
    max_num = n * rate

    root = GBDT.trees[now_tree].root
    left = GBDT.trees[now_tree].root.left
    right = GBDT.trees[now_tree].root.right
    
    for i in range(n):
        now_site = pairs[i].site
        now_score = pairs[i].loss
        if(data[now_site][root.feature] < root.split):
            if(data[now_site][left.feature] < left.split):
                if(number[0] <= max_num):
                    parts[0][now_site] = 1
                    number[0] += 1
                    score[0] += now_score
            else:
                if(number[1] <= max_num):
                    parts[1][now_site] = 1
                    number[1] += 1
                    score[1] += now_score
        else:
            if(data[now_site][right.feature] < right.split):
                if(number[2] <= max_num):
                    parts[2][now_site] = 1
                    number[2] += 1
                    score[2] += now_score
            else:
                if(number[3] <= max_num):
                    parts[3][now_site] = 1
                    number[3] += 1
                    score[3] += now_score
    return(parts, score, number)