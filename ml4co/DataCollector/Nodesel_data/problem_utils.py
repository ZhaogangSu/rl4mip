import os
import sys
import networkx as nx
import random
import pyscipopt as sp
import numpy as np
import multiprocessing as md
from functools import partial

# =================================== GIST ===================================
def dimacsToNx(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if line[0] == 'e':
                g.add_edge(int(arr[1]), int(arr[2]))
    return g

def generateRevsCosts(g, whichSet, setParam):
    if whichSet == 'SET1':
        for node in g.nodes():
            g.nodes[node]['revenue'] = random.randint(1, 100)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = (g.node[u]['revenue'] /
                            + g.node[v]['revenue'])/float(setParam)
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.nodes[node]['revenue'] = float(setParam)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = 1.0


def generateE2(g, alphaE2):
    E2 = set()
    for edge in g.edges():
        if random.random() <= alphaE2:
            E2.add(edge)
    return E2


def createIP(g, E2, ipfilename):
    with open(ipfilename, 'w') as lp_file:
        val = 100
        lp_file.write("maximize\nOBJ:")
        lp_file.write("100x0")
        count = 0
        for node in g.nodes():
            if count:
                lp_file.write(" + " + str(val) + "x" + str(node))
            count += 1
        for edge in E2:
            lp_file.write(" - y" + str(edge[0]) + '_' + str(edge[1]))
        lp_file.write("\n Subject to\n")
        constraint_count = 1
        for node1, node2, edge in g.edges(data=True):
            if (node1, node2) in E2:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+x" + str(node2) + "-y" + str(node1) + "_"
                              + str(node2) + " <=1 \n")
            else:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+" + "x" + str(node2) + " <=1 \n")
            constraint_count += 1

        lp_file.write("\nbinary\n")
        for node in g.nodes():
            lp_file.write(f"x{node}\n")


def generate_sols_setcover(seed_start, seed_end, solve, instances_name_list):

    # print("instances_name_list = ", instances_name_list)
    # 从branch生成的setcover读取实例，然后求解
    for seed in range(seed_start, seed_end):
        instance_name = instances_name_list[seed]
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_name)
            model.optimize()
            instance_name_sol = instance_name.replace("lp", "sol").replace("setcover", "setcover_sol")
            model.writeBestSol(instance_name_sol)

def generate_sols_facilities(seed_start, seed_end, solve, instances_name_list):
    # print(seed_start)
    # print(seed_end)
    # 从branch生成的facilities读取实例，然后求解
    
    for seed in range(seed_start, seed_end):
        instance_name = instances_name_list[seed]
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_name)
            model.optimize()
            instance_name_sol = instance_name.replace("lp", "sol").replace("facilities", "facilities_sol")
            model.writeBestSol(instance_name_sol)


def generate_sols_indset(seed_start, seed_end, solve, instances_name_list):
    # print(seed_start)
    # print(seed_end)
    # 从branch生成的indset读取实例，然后求解
    
    for seed in range(seed_start, seed_end):
        instance_name = instances_name_list[seed]
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_name)
            model.optimize()
            instance_name_sol = instance_name.replace("lp", "sol").replace("indset", "indset_sol")
            model.writeBestSol(instance_name_sol)

def generate_sols_cauctions(seed_start, seed_end, solve, instances_name_list):
    # print(seed_start)
    # print(seed_end)
    # 从branch生成的cauctions读取实例，然后求解
    
    for seed in range(seed_start, seed_end):
        instance_name = instances_name_list[seed]
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_name)
            model.optimize()
            instance_name_sol = instance_name.replace("lp", "sol").replace("cauctions", "cauctions_sol")
            model.writeBestSol(instance_name_sol)

def generate_sols_maxcut(seed_start, seed_end, solve, instances_name_list):
    # print(seed_start)
    # print(seed_end)
    # 从branch生成的 maxcut 读取实例，然后求解
    
    for seed in range(seed_start, seed_end):
        instance_name = instances_name_list[seed]
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_name)
            model.optimize()
            instance_name_sol = instance_name.replace("lp", "sol").replace("maxcut", "maxcut_sol")
            model.writeBestSol(instance_name_sol)


def generate_instances_GISP(seed_start, seed_end, whichSet, setParam, alphaE2, min_n, max_n, er_prob, instance, lp_dir, solve) :
    for seed in range(seed_start, seed_end):
        random.seed(seed)
        if instance is None:
            # Generate random graph
            numnodes = random.randint(min_n, max_n)
            # 生成numnodes个节点，每对节点间有p的概率相连接
            g = nx.erdos_renyi_graph(n=numnodes, p=er_prob, seed=seed)
            lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f_%d"
                    % (numnodes, nx.number_of_edges(g), er_prob, whichSet,
                        setParam, alphaE2, seed))
        else:
            g = dimacsToNx(instance)
            # instanceName = os.path.splitext(instance)[1]
            instanceName = instance.split('/')[-1]
            lpname = ("%s_%s_%g_%g_%d" % (instanceName, whichSet, alphaE2,
                    setParam, seed))
        
        # Generate node revenues and edge costs
        generateRevsCosts(g, whichSet, setParam)
        # Generate the set of removable edges
        E2 = generateE2(g, alphaE2)
        # Create IP, write it to file, and solve it with CPLEX
        #print(lpname)
        # ip = createIP(g, E2, lp_dir + "/" + lpname)
        createIP(g, E2, lp_dir + "/" + lpname + ".lp")
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(lp_dir +"/" + lpname + ".lp")
            model.optimize()
            model.writeBestSol(lp_dir +"/" + lpname + ".sol")



# =================================== FCMFNF ===================================


def get_random_uniform_graph(rng, n_nodes, n_arcs, c_range, d_range, ratio, k_max):
    adj_mat = [[0 for _ in range(n_nodes) ] for _ in range(n_nodes)]
    edge_list = []
    incommings = dict([ (j, []) for j in range(n_nodes) ])
    outcommings = dict([(i, []) for i in range(n_nodes) ])
        
    added_arcs = 0
    #gen network, todo: use 
    while(True):
        i = rng.randint(0,n_nodes) 
        j = rng.randint(0,n_nodes)
        if i ==j or adj_mat[i][j] != 0:
            continue
        else:
            c_ij = rng.uniform(*c_range)
            f_ij = rng.uniform(c_range[0]*ratio, c_range[1]*ratio)
            u_ij = rng.uniform(1,k_max+1)* rng.uniform(*d_range)
            adj_mat[i][j] = (c_ij, f_ij, u_ij)
            added_arcs += 1
            edge_list.append((i,j))
            
            outcommings[i].append(j)
            incommings[j].append(i)

            
            
            
        if added_arcs == n_arcs:
            break
        
    G = nx.DiGraph()
    G.add_nodes_from([i for i in range(n_nodes)])
    G.add_edges_from(edge_list)
    
    return G, adj_mat, edge_list, incommings, outcommings


def get_erdos_graph(rng,n_nodes, c_range, d_range, ratio, k_max, er_prob):
    
    G = nx.erdos_renyi_graph(n=n_nodes, p=er_prob, seed=int(rng.get_state()[1][0]), directed=True)
    adj_mat = [[0 for _ in range(n_nodes) ] for _ in range(n_nodes)]
    edge_list = []
    incommings = dict([ (j, []) for j in range(n_nodes) ])
    outcommings = dict([(i, []) for i in range(n_nodes) ])
    
    for i,j in G.edges:
        c_ij = int(rng.uniform(*c_range))
        f_ij = int(rng.uniform(c_range[0]*ratio, c_range[1]*ratio))
        u_ij = int(rng.uniform(1,k_max+1)* rng.uniform(*d_range))
        adj_mat[i][j] = (c_ij, f_ij, u_ij)
        edge_list.append((i,j))
        
        outcommings[i].append(j)
        incommings[j].append(i)
    
    return G, adj_mat, edge_list, incommings, outcommings
        

def generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob):
    
    G, adj_mat, edge_list, incommings, outcommings = get_erdos_graph(rng, n_nodes, c_range, d_range, ratio, k_max, er_prob)

    # print(G)
     
    commodities = [ 0 for _ in range(n_commodities) ]
    for k in range(n_commodities):
        while True:
            o_k = rng.randint(0, n_nodes)
            d_k = rng.randint(0, n_nodes)
            
            if nx.has_path(G, o_k, d_k) and o_k != d_k:
                break
        
        demand_k = int(rng.uniform(*d_range))
        commodities[k] = (o_k, d_k, demand_k)

    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" + {commodities[k][2]*adj_mat[i][j][0]}x_{i+1}_{j+1}_{k+1}" for (i,j) in edge_list for k in range(n_commodities)]))
        file.write("".join([f" + {adj_mat[i][j][1]}y_{i+1}_{j+1}" for (i,j) in edge_list ]))
        
        
        file.write("\nSubject to\n")
        
        for i in range(n_nodes):
            for k in range(n_commodities):
                
                delta_i = 1 if (commodities[k][0] == i ) else (-1 if commodities[k][1] == i else 0) #1 if source, -1 if sink, 0 if else
                
                file.write(f"flow_{i+1}_{k+1}:" + 
                           "".join([f" +x_{i+1}_{j+1}_{k+1}" for j in outcommings[i] ]) +
                           "".join([f" -x_{j+1}_{i+1}_{k+1}" for j in incommings[i] ]) + f" = {delta_i}\n"   )
                
        
        for (i,j) in edge_list:
            file.write(f"arc_{i+1}_{j+1}:" + 
                       "".join([f" +{commodities[k][2]}x_{i+1}_{j+1}_{k+1}" for k in range(n_commodities) ]) + f"-{adj_mat[i][j][2]}y_{i+1}_{j+1} <= +0\n" )
            

        file.write("\nBinaries\n")
        for (i,j) in edge_list:
            file.write(f" y_{i+1}_{j+1}")
            
        file.write('\nEnd\n')
        file.close()
        
    

def generate_capacited_facility_location(rng, filename, n_customers, n_facilities, ratio=1):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """
    c_x = rng.rand(n_customers)
    c_y = rng.rand(n_customers)

    f_x = rng.rand(n_facilities)
    f_y = rng.rand(n_facilities)

    demands = rng.randint(5, 35+1, size=n_customers)
    capacities = rng.randint(10, 160+1, size=n_facilities)
    fixed_costs = rng.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + rng.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{trans_costs[i, j]}x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
        file.write("".join([f" +{fixed_costs[j]}y_{j+1}" for j in range(n_facilities)]))

        file.write("\nSubject to\n")
        for i in range(n_customers):
            file.write(f"demand_{i+1}:" + "".join([f" -1x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
        for j in range(n_facilities):
            file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]}x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]}y_{j+1} <= +0\n")

        # optional constraints for LP relaxation tightening
        file.write("total_capacity:" + "".join([f" -{capacities[j]}y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f"affectation_{i+1}_{j+1}: +1x_{i+1}_{j+1} -1y_{j+1} <= +0\n")

        file.write("\nBounds\n")
        for i in range(n_customers):
            for j in range(n_facilities):
                file.write(f" 0 <= x_{i+1}_{j+1} <= +1\n")

        file.write("\nBinaries\n")
        for j in range(n_facilities):
            file.write(f" y_{j+1}")
        file.write('\nEnd\n')
        file.close()
    print(filename)

def generate_instances_FCMFNF(start_seed, end_seed, min_n_nodes, max_n_nodes, min_n_commodities, max_n_commodities, er_prob, lp_dir, solveInstance):
    # for seed in range(start_seed, end_seed):
    #     ratio = 5
    #     rng = np.random.RandomState(seed)
    #     instance_id = rng.uniform(0,1)*100

    #     n_nodes =  rng.randint(min_n_nodes, max_n_nodes+1)
    #     n_commodities = rng.randint(min_n_commodities, max_n_commodities+1)

    #     c_range = (11,50)
    #     d_range = (10,100)
        
    #     k_max = n_commodities #loose
    #     ratio = 100

    #     instance_name = f'n_nodes={n_nodes}_n_commodities={n_commodities}_id_{instance_id:0.2f}'
        
    #     instance_path = lp_dir +  "/" + instance_name
    #     filename = instance_path+'.lp'

    #     # print(instance_name)

    #     generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob)
        
    #     model = sp.Model()
    #     model.hideOutput()
    #     model.readProblem(instance_path + ".lp")
        
    #     if solveInstance:
    #         model.optimize()
    #         model.writeBestSol(instance_path + ".sol")  
    #         print(model.getNNodes())
            
    #         if model.getNNodes() <= 1:
    #             os.remove(instance_path+ ".lp" )
    #             os.remove(instance_path+ ".sol")
    #             # 重新执行本次循环，不要浪费训练次数


    # # 修改代码，防止model.getNNodes() <= 1时删除了该问题，导致生成的instance小于预期
    # for seed in range(start_seed, end_seed):
    #     ratio = 5
    #     rng = np.random.RandomState(seed)
    #     instance_id = rng.uniform(0, 1) * 100

    #     n_nodes = rng.randint(min_n_nodes, max_n_nodes + 1)
    #     n_commodities = rng.randint(min_n_commodities, max_n_commodities + 1)

    #     c_range = (11, 50)
    #     d_range = (10, 100)

    #     k_max = n_commodities  # loose
    #     ratio = 100

    #     instance_name = f'n_nodes={n_nodes}_n_commodities={n_commodities}_id_{instance_id:0.2f}'
        
    #     instance_path = lp_dir + "/" + instance_name
    #     filename = instance_path + '.lp'

    #     # print(instance_name)

    #     generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob)
        
    #     model = sp.Model()
    #     model.hideOutput()
    #     model.readProblem(instance_path + ".lp")
        
    #     # Retry mechanism for solving the instance
    #     while True:
    #         if solveInstance:
    #             model.optimize()
    #             model.writeBestSol(instance_path + ".sol")  
    #             print(model.getNNodes())

    #             if model.getNNodes() <= 1:
    #                 # If solution is invalid, remove the generated files and retry
    #                 os.remove(instance_path + ".lp")
    #                 os.remove(instance_path + ".sol")
    #                 print(f"Invalid solution for {instance_name}. Retrying...")
    #                 # Re-generate the instance and retry solving it
    #                 generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob)
    #                 model.readProblem(instance_path + ".lp")
    #             else:
    #                 # If solution is valid, break out of the retry loop
    #                 break


    # for seed in range(start_seed, end_seed):
    #     retry_count = 0
        
    #     while True:
    #         # 使用修改后的种子确保每次重试生成不同的实例
    #         current_seed = seed if retry_count == 0 else seed * 1000 + retry_count
            
    #         ratio = 5
    #         rng = np.random.RandomState(current_seed)
    #         instance_id = rng.uniform(0,1)*100
    #         n_nodes =  rng.randint(min_n_nodes, max_n_nodes+1)
    #         n_commodities = rng.randint(min_n_commodities, max_n_commodities+1)
    #         c_range = (11,50)
    #         d_range = (10,100)
            
    #         k_max = n_commodities #loose
    #         ratio = 100
    #         instance_name = f'n_nodes={n_nodes}_n_commodities={n_commodities}_id_{instance_id:0.2f}'
            
    #         instance_path = lp_dir +  "/" + instance_name
    #         filename = instance_path+'.lp'
    #         # print(instance_name)
    #         generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob)
            
    #         model = sp.Model()
    #         model.hideOutput()
    #         model.readProblem(instance_path + ".lp")
            
    #         if solveInstance:
    #             model.optimize()
    #             model.writeBestSol(instance_path + ".sol")  
    #             print(model.getNNodes())
                
    #             if model.getNNodes() <= 1:
    #                 os.remove(instance_path+ ".lp" )
    #                 os.remove(instance_path+ ".sol")
    #                 retry_count += 1
    #                 print(f"实例过于简单(model.getNNodes() <= 1)，重试第 {retry_count} 次")
    #                 # 重新执行本次循环
    #                 continue
    #             else:
    #                 # 实例有效，退出重试循环
    #                 break
    #         else:
    #             # 不求解的话直接退出
    #             break


    for seed in range(start_seed, end_seed):
        retry_count = 0
        
        while True:
            # 使用修改后的种子确保每次重试生成不同的实例
            current_seed = seed if retry_count == 0 else seed * 1000 + retry_count
            
            ratio = 5
            rng = np.random.RandomState(current_seed)
            instance_id = rng.uniform(0,1)*100
            n_nodes =  rng.randint(min_n_nodes, max_n_nodes+1)
            n_commodities = rng.randint(min_n_commodities, max_n_commodities+1)
            c_range = (11,50)
            d_range = (10,100)
            
            k_max = n_commodities #loose
            ratio = 100
            instance_name = f'n_nodes={n_nodes}_n_commodities={n_commodities}_id_{instance_id:0.2f}'
            
            instance_path = lp_dir +  "/" + instance_name
            filename = instance_path+'.lp'
            
            # 检查文件是否已存在，如果存在则重新生成
            if os.path.exists(filename):
                retry_count += 1
                print(f"文件已存在({filename})，重新生成第 {retry_count} 次")
                continue
                
            # print(instance_name)
            generate_fcmcnf(rng, filename, n_nodes, n_commodities, c_range, d_range, k_max, ratio, er_prob)
            
            model = sp.Model()
            model.hideOutput()
            model.readProblem(instance_path + ".lp")
            
            if solveInstance:
                model.optimize()
                model.writeBestSol(instance_path + ".sol")  
                print(model.getNNodes())
                
                if model.getNNodes() <= 1:
                    os.remove(instance_path+ ".lp" )
                    os.remove(instance_path+ ".sol")
                    retry_count += 1
                    print(f"实例过于简单(model.getNNodes() <= 1)，重试第 {retry_count} 次")
                    # 重新执行本次循环
                    continue
                else:
                    # 实例有效，退出重试循环
                    break
            else:
                # 不求解的话直接退出
                break









# =================================== WPMS ===================================

def get_bipartite(n1,n2,p):
    nx.bipartite_random_graph
    

def gen_maxcut_graph_clauses(rng,n,er_prob,p=0.3):
    
    G = nx.erdos_renyi_graph(n=n, p=er_prob, seed=int(rng.get_state()[1][0]), directed=True) 
    
    divider = rng.randint(1,6)
    G = nx.algorithms.bipartite.generators.random_graph(n//divider, n - n//divider, p=er_prob, seed=int(rng.get_state()[1][0]))
    
    n_edges = len(G.edges)
    edges = list(G.edges)
    
    
    added_edges = 0
    while added_edges < n_edges*p:
        i,j = rng.randint(0,n), rng.randint(0,n)
        if (i,j) not in edges and (j,i) not in edges:
            added_edges += 1 
            edges.append((i,j))
        
    return [ ( f'v{i},v{j}', 1 )  for (i,j) in edges ] +  [ (f'-v{i},-v{j}', 1) for (i,j) in edges ]
            

#Ramon Bejar
#https://computational-sustainability.cis.cornell.edu/cpaior2013/pdfs/ansotegui.pdf
def get_clauses(rng,H,W, n_piece, n_obstacle):

    pieces = [(rng.randint(1,H+1), rng.randint(1,W+1)) for _ in range(n_piece) ] + [(1,1) ]*n_obstacle
    
    #Generate obstacles, aka filled pieces
    obstacle_pos = []
    while(len(obstacle_pos) != n_obstacle):
        candidat = (rng.randint(0,H), rng.randint(0,W))
        if candidat not in obstacle_pos:
            obstacle_pos.append(candidat)  
    
    #Soft clauses
    clauses = [(f'x{k}', piece[0]*piece[1]) for k,piece in enumerate(pieces[:n_piece])]

    #Hard clauses
    #No  two times same row/colomn per piece
    for k,piece in enumerate(pieces):
        h,w = piece
        clauses.append((f'-x{k},' + ','.join([ f'r{i}_{k}' for i in range(0,H-h+1)  ]),np.inf))

        for i in range(0, H-h+1):
            for j in range(i+1, H-h+1):
                clauses.append((f'-x{k},-r{i}_{k},-r{j}_{k}',np.inf))

        
        clauses.append((f'-x{k},' + ','.join([ f'c{j}_{k}' for j in range(0,W-w+1)  ]), np.inf))

        for i in range(0, W-w+1):
            for j in range(i+1,W-w+1):
                clauses.append((f'-x{k},-c{i}_{k},-c{j}_{k}',np.inf))

    #Place obstacles
    for k,pos in enumerate(obstacle_pos):
        i,j = pos
        reajusted_k = n_piece + k
        clauses.append((f'x{reajusted_k}', np.inf))
        clauses.append((f'r{i}_{reajusted_k}', np.inf))
        clauses.append((f'c{j}_{reajusted_k}', np.inf))

    # #No overlapping between pieces(and obstacles)

    cl_s  = []
    for s,piece1 in enumerate( pieces ):
        h,w = piece1

        for k,piece2 in enumerate(pieces):
            if k != s:
    
                for i in range(0, H-h+1):
                    for j in range(0, W-w+1):
                        for l in range(i, i+h):
                            for m in range(j, j+w):
    
                                cl = (f'-r{i}_{s},-c{j}_{s},-r{l}_{k},-c{m}_{k}',np.inf)
                                cl_s.append(cl)
            
    return clauses + cl_s



def write_lp(clauses, filename):
    
    ''' 
        clauses (in conj normal form )  : list of clauses to be "and-ed" with their weiths
        
        Clause  : string representing a conjunctive (or's') clause, variable seperated by ',', 
        negation of variable  represented by -.
        
        
        
        Ex : 2*(A1 or not(A2)) and 1*(not(C)) == [ ('A,-A2', 2) , ('-C',1) ]
        
        '''
    var_names = dict() #maps var appearing in order i in clauses (whatever the name ) to y_i

    with open(filename, 'w') as file:
        file.write("maximize\nOBJ:")
        file.write("".join([f" +{clause[1]}cl_{idx}" for idx,clause in enumerate(clauses) if clause[1] < np.inf ]))

        
        
        file.write("\n\nSubject to\n")
    
        for idx,clause in enumerate(clauses):
            varrs = clause[0].split(',')
            
            neg_varrs = []
            pos_varrs = []
            
            for var in varrs:
                if var != '':
                    if var[0] == '-':
                        if var[1:] not in var_names:
                            var_names[var[1:]] =  var[1:]
                        neg_varrs.append(var_names[var[1:]])
                        
                    else:
                        if var[0:] not in var_names:
                            var_names[var[0:]] = var[0:]
                        pos_varrs.append(var_names[var[0:]])
                        
                            
            last_part = f' +cl_{idx} <= {len(neg_varrs)} \n' if clause[1] < np.inf else f' <= {len(neg_varrs) - 1} \n'           
                    
            
            file.write(f"clause_{idx}:" + ''.join([ f" -{yi}" for yi in pos_varrs]) + ''.join([ f" +{yi}" for yi in neg_varrs]) + last_part) 
                       
        file.write("\nBinaries\n")
        for idx in range(len(clauses)):
            if clauses[idx][1] < np.inf:
                file.write(f" cl_{idx}")
            
        for var_name in var_names.keys():
            file.write(f" {var_names[var_name]}")
            
        file.write('\nEnd\n')
        file.close()


def generate_instances_WPMS(start_seed, end_seed, min_n, max_n, lp_dir, solveInstance, er_prob):
    
    # for seed in range(start_seed, end_seed):
        
    #     rng = np.random.RandomState(seed)
    #     instance_id = rng.uniform(0,1)*100
        
        
    #     n = rng.randint(min_n, max_n+1)
    #     clauses = gen_maxcut_graph_clauses(rng,n,er_prob)
    #     m = len(clauses)//2

    # 修改代码，防止m=0的时候，不会生成instance，导致最终生成的数量小于预期的
    for seed in range(start_seed, end_seed):
        
        rng = np.random.RandomState(seed)
        instance_id = rng.uniform(0, 1) * 100
        
        while True:
            n = rng.randint(min_n, max_n + 1)
            clauses = gen_maxcut_graph_clauses(rng, n, er_prob)
            m = len(clauses) // 2
            
            if m != 0:
                break  # Exit the loop if m is non-zero
            
        # Continue with the rest of the code once valid m is found
        # Your additional logic here


        instance_name = f'n={n}_m={m}_id_{instance_id:0.2f}'

        instance_path = lp_dir +  "/" + instance_name
        write_lp(clauses, instance_path + ".lp")
        print(instance_name)
        
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance_path + ".lp")
        
        if solveInstance:
            model.optimize()
            model.writeBestSol(instance_path + ".sol")  
            # print(model.getNNodes())
            # print(model.getSolvingTime())
            
            if model.getNNodes() <= 1:
                os.remove(instance_path+ ".lp" )
                os.remove(instance_path+ ".sol")