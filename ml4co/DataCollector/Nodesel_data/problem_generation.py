import os
import sys
import networkx as nx
import random
import numpy as np
import multiprocessing as md
from functools import partial
from pathlib import Path 

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem_utils import generate_instances_GISP, generate_instances_FCMFNF, generate_instances_WPMS
from problem_utils import generate_sols_setcover, generate_sols_facilities, generate_sols_indset, generate_sols_cauctions, generate_sols_maxcut

'''
Used for generating GISP, FCMCNF, WPMS
'''

def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res


def problem_generator(problem='GISP', ntrain=10, nvalid=10, ntest=10, datapath=None, n_cpu=16):
    if problem == 'GISP':
        if ntrain:
            GISP_generator(n_instance=ntrain, data_partition='train', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if nvalid:
            GISP_generator(n_instance=nvalid, data_partition='valid', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if ntest:
            GISP_generator(n_instance=ntest, data_partition='test', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

            GISP_generator(n_instance=ntest, data_partition='small_test', instance=None, n_cpu=n_cpu,
                    max_n=80, min_n=70, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

            GISP_generator(n_instance=ntest, data_partition='medium_test', instance=None, n_cpu=n_cpu,
                    max_n=90, min_n=80, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

            GISP_generator(n_instance=ntest, data_partition='big_test', instance=None, n_cpu=n_cpu,
                    max_n=100, min_n=90, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
    
    elif problem == 'FCMCNF':
        if ntrain:
            FCMFNF_generator(n_instance=ntrain, data_partition='train', instance=None, n_cpu=n_cpu,
                    max_n=15, min_n=15, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if nvalid:
            FCMFNF_generator(n_instance=nvalid, data_partition='valid', instance=None, n_cpu=n_cpu,
                    max_n=15, min_n=15, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if ntest:
            # FCMFNF_generator(n_instance=ntest, data_partition='test', instance=None, n_cpu=n_cpu,
            #         max_n=15, min_n=15, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        
            FCMFNF_generator(n_instance=ntest, data_partition='small_test', instance=None, n_cpu=n_cpu,
                    max_n=15, min_n=15, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        
            FCMFNF_generator(n_instance=ntest, data_partition='medium_test', instance=None, n_cpu=n_cpu,
                    max_n=20, min_n=20, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
            
            FCMFNF_generator(n_instance=ntest, data_partition='big_test', instance=None, n_cpu=n_cpu,
                    max_n=25, min_n=25, er_prob=0.33, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

    elif problem == 'WPMS':
        if ntrain:

            WPMS_generator(n_instance=ntrain, data_partition='train', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if nvalid:
        
            WPMS_generator(n_instance=nvalid, data_partition='valid', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        if ntest:
        
            WPMS_generator(n_instance=ntest, data_partition='test', instance=None, n_cpu=n_cpu,
                    max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)
        
            WPMS_generator(n_instance=ntest, data_partition='small_test', instance=None, n_cpu=n_cpu,
                    max_n=80, min_n=70, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

            WPMS_generator(n_instance=ntest, data_partition='medium_test', instance=None, n_cpu=n_cpu,
                    max_n=90, min_n=80, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)

            WPMS_generator(n_instance=ntest, data_partition='big_test', instance=None, n_cpu=n_cpu,
                    max_n=100, min_n=90, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=datapath)


def sol_generator(problem='setcover', ntrain=10, nvalid=10, ntest=10, datapath=None, n_cpu=16):
    if problem == 'setcover':
        if ntrain:
            setcover_generator(n_instance=ntrain, data_partition='train', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if nvalid:
            setcover_generator(n_instance=nvalid, data_partition='valid', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if ntest:
            setcover_generator(n_instance=ntest, data_partition='small_test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
            setcover_generator(n_instance=ntest, data_partition='medium_test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
            setcover_generator(n_instance=ntest, data_partition='big_test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
    
    elif problem == 'indset':
        if ntrain:
            indset_generator(n_instance=ntrain, data_partition='train', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if nvalid:
            indset_generator(n_instance=nvalid, data_partition='valid', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if ntest:
            indset_generator(n_instance=ntest, data_partition='test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
    elif problem == 'cauctions':
        if ntrain:
            cauctions_generator(n_instance=ntrain, data_partition='train', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if nvalid:
            cauctions_generator(n_instance=nvalid, data_partition='valid', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if ntest:        
            cauctions_generator(n_instance=ntest, data_partition='test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
    elif problem == 'facilities':
        if ntrain:
            facilities_generator(n_instance=ntrain, data_partition='train', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if nvalid:        
            facilities_generator(n_instance=nvalid, data_partition='valid', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if ntest:       
            facilities_generator(n_instance=ntest, data_partition='test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
    elif problem == 'maxcut':
        if ntrain:
            maxcut_generator(n_instance=ntrain, data_partition='train', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if nvalid:
            maxcut_generator(n_instance=nvalid, data_partition='valid', n_cpu=n_cpu, solveInstance=True, datapath=datapath)
        if ntest:
            maxcut_generator(n_instance=ntest, data_partition='test', n_cpu=n_cpu, solveInstance=True, datapath=datapath)


def setcover_generator(n_instance, data_partition='train', n_cpu=16, solveInstance=True, datapath=None):
    '''setcover'''
    seed = random.randint(0, 2**32 - 1)
    lp_dir = os.path.join(datapath, 'instances', 'setcover', data_partition)
    sol_dir = os.path.join(datapath, 'instances', 'setcover_sol', data_partition)

    try:
        os.makedirs(sol_dir)
    except FileExistsError:
        ""

    print(f"Summary for setcover {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"lp dir    :     {lp_dir}")
    print(f"sol dir    :     {sol_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    instances_name_list = [os.path.abspath(os.path.join(lp_dir, f)) for f in os.listdir(lp_dir) if os.path.isfile(os.path.join(lp_dir, f))]
    random.shuffle(instances_name_list)
    instances_name_list = instances_name_list[:n_instance]

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_sols_setcover,
                                                                      p1, 
                                                                      p2,
                                                                      solveInstance,
                                                                      instances_name_list))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')



def indset_generator(n_instance, data_partition='train', n_cpu=16, solveInstance=True, datapath=None):
    '''indset'''
    seed = random.randint(0, 2**32 - 1)
    lp_dir = os.path.join(datapath, 'instances', 'indset', data_partition)
    sol_dir = os.path.join(datapath, 'instances', 'indset_sol', data_partition)

    try:
        os.makedirs(sol_dir)
    except FileExistsError:
        ""

    print(f"Summary for indset {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"lp dir    :     {lp_dir}")
    print(f"sol dir    :     {sol_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    instances_name_list = [os.path.abspath(os.path.join(lp_dir, f)) for f in os.listdir(lp_dir) if os.path.isfile(os.path.join(lp_dir, f))]
    random.shuffle(instances_name_list)
    instances_name_list = instances_name_list[:n_instance]

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_sols_indset,
                                                                      p1, 
                                                                      p2, 
                                                                      solveInstance,
                                                                      instances_name_list))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')


def cauctions_generator(n_instance, data_partition='train', n_cpu=16, solveInstance=True, datapath=None):
    '''cauctions'''
    seed = random.randint(0, 2**32 - 1)
    lp_dir = os.path.join(datapath, 'instances', 'cauctions', data_partition)
    sol_dir = os.path.join(datapath, 'instances', 'cauctions_sol', data_partition)

    try:
        os.makedirs(sol_dir)
    except FileExistsError:
        ""

    # try:
    #     os.makedirs(lp_dir)
    # except FileExistsError:
    #     ""
    print(f"Summary for cauctions {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    # print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"lp dir    :     {lp_dir}")
    print(f"sol dir    :     {sol_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    instances_name_list = [os.path.abspath(os.path.join(lp_dir, f)) for f in os.listdir(lp_dir) if os.path.isfile(os.path.join(lp_dir, f))]
    random.shuffle(instances_name_list)
    instances_name_list = instances_name_list[:n_instance]

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_sols_cauctions,
                                                                      p1, 
                                                                      p2, 
                                                                      solveInstance,
                                                                      instances_name_list))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')

def facilities_generator(n_instance, data_partition='train', n_cpu=16, solveInstance=True, datapath=None):
    '''facilities'''
    seed = random.randint(0, 2**32 - 1)

    lp_dir = os.path.join(datapath, 'instances', 'facilities', data_partition)
    sol_dir = os.path.join(datapath, 'instances', 'facilities_sol', data_partition)

    try:
        os.makedirs(sol_dir)
    except FileExistsError:
        ""

    # try:
    #     os.makedirs(lp_dir)
    # except FileExistsError:
    #     ""
    print(f"Summary for facilities {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    # print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"lp dir    :     {lp_dir}")
    print(f"sol dir    :     {sol_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    instances_name_list = [os.path.abspath(os.path.join(lp_dir, f)) for f in os.listdir(lp_dir) if os.path.isfile(os.path.join(lp_dir, f))]
    random.shuffle(instances_name_list)
    instances_name_list = instances_name_list[:n_instance]

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_sols_facilities,
                                                                      p1, 
                                                                      p2, 
                                                                      solveInstance,
                                                                      instances_name_list))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')


def maxcut_generator(n_instance, data_partition='train', n_cpu=16, solveInstance=True, datapath=None):
    '''maxcut'''
    seed = random.randint(0, 2**32 - 1)
    lp_dir = os.path.join(datapath, 'instances', 'maxcut', data_partition)
    sol_dir = os.path.join(datapath, 'instances', 'maxcut_sol', data_partition)
    
    try:
        os.makedirs(sol_dir)
    except FileExistsError:
        ""

    # try:
    #     os.makedirs(lp_dir)
    # except FileExistsError:
    #     ""
    print(f"Summary for maxcut {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    # print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"lp dir    :     {lp_dir}")
    print(f"sol dir    :     {sol_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    instances_name_list = [os.path.abspath(os.path.join(lp_dir, f)) for f in os.listdir(lp_dir) if os.path.isfile(os.path.join(lp_dir, f))]
    random.shuffle(instances_name_list)
    instances_name_list = instances_name_list[:n_instance]

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_sols_maxcut,
                                                                      p1, 
                                                                      p2, 
                                                                      solveInstance,
                                                                      instances_name_list))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')


def GISP_generator(n_instance, data_partition='train', seed=0, instance=None, n_cpu=16,
                max_n=70, min_n=60, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=None):
    '''GISP'''

    seed = random.randint(0, 2**32 - 1)
    seed = 0
    # exp_dir = datapath + data_partition
    exp_dir = os.path.join(datapath, 'instances')
    exp_dir = os.path.join(exp_dir, 'GISP')
    lp_dir = os.path.join(exp_dir, data_partition)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""

    print(f"Summary for GIST {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_instances_GISP,
                                                                      seed + p1, 
                                                                      seed + p2, 
                                                                      whichSet, 
                                                                      setParam, 
                                                                      alphaE2, 
                                                                      min_n, 
                                                                      max_n, 
                                                                      er_prob, 
                                                                     instance, 
                                                                      lp_dir, 
                                                                      solveInstance))
                     for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')

def FCMFNF_generator(n_instance, data_partition='train', seed=0, instance=None, n_cpu=16,
                max_n=15, min_n=15, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=None):
    '''FCMCNF'''
    seed = random.randint(0, 2**32 - 1)
    seed = 0
    #number of commodities for FCMCNF
    min_n_commodities = max_n
    max_n_commodities = int(1.5*max_n)

    # exp_dir = datapath + data_partition
    exp_dir = os.path.join(datapath, 'instances')
    exp_dir = os.path.join(exp_dir, 'FCMCNF')
    lp_dir = os.path.join(exp_dir, data_partition)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""

    print(f"Summary for FCMCNF {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu

    print(distribute(n_instance, n_cpu))

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_instances_FCMFNF,
                                                                      seed + p1, 
                                                                      seed + p2, 
                                                                      min_n,
                                                                      max_n,
                                                                      min_n_commodities,
                                                                      max_n_commodities,
                                                                      er_prob,
                                                                      lp_dir, 
                                                                      solveInstance))
                      for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')

def WPMS_generator(n_instance, data_partition='train', seed=0, instance=None, n_cpu=16,
                max_n=15, min_n=15, er_prob=0.6, whichSet='SET2', setParam=100.0, alphaE2=0.5, timelimit=3600.0, solveInstance=True, datapath=None):
    '''WPMS'''
    seed = random.randint(0, 2**32 - 1)
    seed = 0
    # exp_dir = datapath + data_partition
    exp_dir = os.path.join(datapath, 'instances')
    exp_dir = os.path.join(exp_dir, 'WPMS')
    lp_dir = os.path.join(exp_dir, data_partition)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""

    print(f"Summary for WPMS {data_partition} generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")

    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu

    print(distribute(n_instance, n_cpu))

    processes = [  md.Process(name=f"worker {p}", target=partial(generate_instances_WPMS,
                                                                      seed + p1, 
                                                                      seed + p2, 
                                                                      min_n,
                                                                      max_n,
                                                                      lp_dir, 
                                                                      solveInstance,
                                                                      er_prob=er_prob))
                      for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    
    print('Generated')