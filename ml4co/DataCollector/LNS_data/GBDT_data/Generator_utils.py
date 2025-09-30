import numpy as np
import random
import os

def generate_IS(N, M):
    '''
    Function Description:
    Generate instances of the maximum independent set problem in a general graph.
    
    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated maximum independent set problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    # Add constraint: randomly generate an edge and impose a constraint that the vertices connected by the edge cannot be selected simultaneously.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        site[i].append(x)
        value[i].append(1)
        site[i].append(y) 
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 1
        k[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
        lower_bound[i] = 0
        upper_bound[i] = 1
        value_type[i] = "Binary"
    return(n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type)


def generate_MVC(N, M):
    '''
    Function Description:
    Generate instances of the minimum vertex cover problem in a general graph.

    Parameters:
    - N: Number of vertices in the graph.
    - M: Number of edges in the graph.

    Return: 
    Relevant parameters of the generated minimum vertex cover problem.
    '''
    
    # n represents the number of decision variables, where each vertex in the graph corresponds to a decision variable.
    # m represents the number of constraints, where each edge in the graph corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    # Add constraint: randomly generate an edge and impose a constraint that at least one of the vertices connected by the edge must be selected.
    for i in range(M):
        x = random.randint(0, N - 1)
        y = random.randint(0, N - 1)
        while(x == y) :
            x = random.randint(0, N - 1)
            y = random.randint(0, N - 1)
        k[i] = 2
        site[i].append(x)
        value[i].append(1)
        site[i].append(y)
        value[i].append(1)
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a vertex is a random value.
    for i in range(N):
        coefficient[i] = random.random()
        lower_bound[i] = 0
        upper_bound[i] = 1
        value_type[i] = "Binary"
    return(n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type)

def generate_SC(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 4 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''

    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    # Add constraint: At least one of the four sets in which each item appears must be selected.
    for i in range(M):
        vis = {}
        for j in range(4):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 4   
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 2
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random()
        lower_bound[i] = 0
        upper_bound[i] = 1
        value_type[i] = "Binary"
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type)

def generate_CA(N, M):
    '''
    Function Description:
    Generate instances of the set cover problem, where each item is guaranteed to appear in exactly 5 sets.

    Parameters:
    - N: Number of sets.
    - M: Number of items.

    Return: 
    Relevant parameters of the generated set cover problem.
    '''
    # n represents the number of decision variables, where each set corresponds to a decision variable.
    # m represents the number of constraints, where each item corresponds to a constraint.
    # k[i] represents the number of decision variables in the i-th constraint.
    n = N
    m = M
    k = []

    # site[i][j] represents which decision variable the j-th decision variable corresponds to in the i-th constraint.
    # value[i][j] represents the coefficient of the j-th decision variable in the i-th constraint.
    # constraint[i] represents the right-hand side value of the i-th constraint.
    # constraint_type[i] represents the type of the i-th constraint, where 1 represents <=, 2 represents >=, and 3 represents =.
    # coefficient[i] represents the coefficient of the i-th decision variable in the objective function.
    
    site = []
    value = []
    for i in range(m):
        site.append([])
        value.append([])
        k.append(0)
    constraint = np.zeros(m)
    constraint_type = np.zeros(m)
    coefficient = {}
    lower_bound = {}
    upper_bound = {}
    value_type = {}
    # Add constraints.
    for i in range(M):
        vis = {}
        for j in range(5):
            now = random.randint(0, N - 1)
            while(now in vis.keys()):
                now = random.randint(0, N - 1)
            vis[now] = 1

            site[i].append(now)
            value[i].append(1)
        k[i] = 5   
    for i in range(M):
        constraint[i] = 1
        constraint_type[i] = 1
    
    # Set the coefficients of the objective function, where the coefficient value of each decision variable corresponding to a set is a random value.
    for i in range(N):
        coefficient[i] = random.random() * 1000
        lower_bound[i] = 0
        upper_bound[i] = 1
        value_type[i] = "Binary"
    
    return(n, m, k, site, value, constraint, constraint_type, coefficient, lower_bound, upper_bound, value_type)

def write_lp_file_Max(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, i):
    
    filename = os.path.join(lp_dir, f'instance_{i}.lp')

    with open(filename, 'w') as f:
        f.write("Maximize\n")
        f.write("obj: ")
        for i in range(n):
            if i in coefficient:
                f.write(f"{coefficient[i]:.6f} x{i} + ")
        f.write("0\n")  
        
        f.write("Subject To\n")
        for i in range(m):
            f.write(f"c{i}: ")
            for j in range(k[i]):
                if j == k[i]-1:
                    f.write(f"{value[i][j]} x{site[i][j]}")
                else:
                    f.write(f"{value[i][j]} x{site[i][j]} + ")
            if constraint_type[i] == 1:  # <=
                f.write(f" <= {constraint[i]}\n")
            elif constraint_type[i] == 2:  # >=
                f.write(f" >= {constraint[i]}\n")
            elif constraint_type[i] == 3:  # =
                f.write(f" = {constraint[i]}\n")
        
        f.write("Bounds\n")

        for i in range(n):
            f.write(f"0 <= x{i} <= 1\n")


        f.write("Binary\n")
        for i in range(n):
            f.write(f"x{i}\n")
        
        f.write("End\n")

def write_lp_file_Min(lp_dir, n, m, k, site, value, constraint, constraint_type, coefficient, i):
    
    filename = os.path.join(lp_dir, f'instance_{i}.lp')
    
    with open(filename, 'w') as f:
        f.write("Minimize\n")
        f.write("obj: ")
        for i in range(n):
            if i in coefficient:
                f.write(f"{coefficient[i]:.6f} x{i} + ")
        f.write("0\n")  
        
        f.write("Subject To\n")
        for i in range(m):
            f.write(f"c{i}: ")
            for j in range(k[i]):
                if j == k[i]-1:
                    f.write(f"{value[i][j]} x{site[i][j]}")
                else:
                    f.write(f"{value[i][j]} x{site[i][j]} + ")
            if constraint_type[i] == 1:  # <=
                f.write(f" <= {constraint[i]}\n")
            elif constraint_type[i] == 2:  # >=
                f.write(f" >= {constraint[i]}\n")
            elif constraint_type[i] == 3:  # =
                f.write(f" = {constraint[i]}\n")
        
        f.write("Bounds\n")

        for i in range(n):
            f.write(f"0 <= x_{i} <= 1\n")

        f.write("Binary\n")
        for i in range(n):
            f.write(f"x{i}\n")
        
        f.write("End\n")