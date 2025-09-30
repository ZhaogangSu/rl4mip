import os
import pickle
import glob
import gzip
import shutil
import multiprocessing as mp
import numpy as np
import pyscipopt as scip


def init_scip_params(model, seed, heuristics=True, presolving=True, separating=True, conflict=True):

    seed = seed % 2147483648  # SCIP seed range

    # set up randomization
    model.setBoolParam('randomization/permutevars', True)
    model.setIntParam('randomization/permutationseed', seed)
    model.setIntParam('randomization/randomseedshift', seed)

    # separation only at root node
    model.setIntParam('separating/maxrounds', 0)

    # no restart
    model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable presolving
    if not presolving:
        model.setIntParam('presolving/maxrounds', 0)
        model.setIntParam('presolving/maxrestarts', 0)

    # if asked, disable separating (cuts)
    if not separating:
        model.setIntParam('separating/maxroundsroot', 0)

    # if asked, disable conflict analysis (more cuts)
    if not conflict:
        model.setBoolParam('conflict/enable', False)

    # if asked, disable primal heuristics
    if not heuristics:
        model.setHeuristics(scip.SCIP_PARAMSETTING.OFF)


class VanillaFullstrongBranchingDataCollector(scip.Branchrule):
    """
    Implements branching policy to be used by SCIP such that data collection required for hybrid models is embedded in it.
    """
    def __init__(self, rng, query_expert_prob=0.6):
        self.khalil_root_buffer = {}
        self.obss = []
        self.targets = []
        self.obss_feats = []
        self.exploration_policy = "pscost"
        self.query_expert_prob = query_expert_prob
        self.rng = rng
        self.iteration_counter = 0

    def branchinit(self):
        self.ndomchgs = 0
        self.ncutoffs = 0
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):
        self.iteration_counter += 1

        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert or self.model.getNNodes() == 1:
            candidate_vars, *_ = self.model.getPseudoBranchCands()
            candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]

            var_features, edge_features, cons_features, _ = self.model.getBipartiteGraphRepresentation()
            candi_features, _, _ = self.model.getBranchFeaturesRepresentation(candidate_vars)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            best_var = cands_[bestcand]

            self.add_obs(best_var, (var_features, edge_features, cons_features, candi_features), (cands_, scores), candidate_mask)
            if self.model.getNNodes() == 1:
                self.state = [(var_features, edge_features, cons_features), candi_features]

            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
        else:
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result':result}

    def add_obs(self, best_var, state_, cands_scores=None, candidate_mask=None):
        """
        Adds sample to the `self.obs` to be processed later at the end of optimization.

        Parameters
        ----------
            best_var : pyscipopt.Variable
                object representing variable in LP
            state_ : tuple
                extracted features of constraints and variables at a node
            cands_scores : np.array
                scores of each of the candidate variable on which expert policy was executed

        Return
        ------
        (bool): True if sample is added succesfully. False o.w.
        """
        if self.model.getNNodes() == 1:
            self.obss = []
            self.targets = []
            self.obss_feats = []
            self.map = sorted([x.getCol().getIndex() for x in self.model.getVars(transformed=True)])

        cands, scores = cands_scores
        # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
        if any([s < 0 for s in scores]):
            return False

        var_features, edge_features, cons_features, candi_features = state_

        indices = [[row[1] for row in edge_features],[row[0] for row in edge_features]]
        values = [row[2] for row in edge_features]
        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        variance = sum(squared_diffs) / len(squared_diffs)
        std = variance ** 0.5 + 1e-4
        normalized_values = [(x - mean) / std for x in values]
        edge_features_dic = {'indices':indices, 'values':normalized_values}

        # add more
        cands_index = [x.getCol().getIndex() for x in cands]
        tmp_scores = -np.ones(len(self.map))
        if scores:
            tmp_scores[cands_index] = scores

        self.targets.append(best_var.getCol().getIndex())
        self.obss.append([var_features, cons_features, edge_features_dic, candi_features])
        depth = self.model.getCurrentNode().getDepth()
        self.obss_feats.append({'depth':depth, 'action_set': candidate_mask, 'scores':np.array(tmp_scores), 'iteration': self.iteration_counter})

        return True

def make_samples(in_queue, out_queue):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        Input queue from which orders are received.
    out_queue : multiprocessing.Queue
        Output queue in which to send samples.
    """
    while True:
        episode, instance, seed, time_limit, outdir, rng = in_queue.get()

        m = scip.Model()
        m.setIntParam('display/verblevel', 0)
        m.readProblem(f'{instance}')
        init_scip_params(m, seed=seed)
        m.setIntParam('timing/clocktype', 2)
        m.setRealParam('limits/time', time_limit)

        branchrule = VanillaFullstrongBranchingDataCollector(rng)
        m.includeBranchrule(
            branchrule=branchrule,
            name="Sampling branching rule", desc="",
            priority=666666, maxdepth=-1, maxbounddist=1)

        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

        out_queue.put({
            "type":'start',
            "episode":episode,
            "instance":instance,
            "seed": seed
        })

        m.optimize()
        # data storage - root and node data are saved separately.
        # node data carries a reference to the root filename.
        if m.getNNodes() >= 1 and len(branchrule.obss) > 0 :
            filenames = []
            max_depth = max(x['depth'] for x in branchrule.obss_feats)
            stats = {'nnodes':m.getNNodes(), 'time':m.getSolvingTime(), 'gap':m.getGap(), 'nobs':len(branchrule.obss)}

            # prepare root data
            sample_state, sample_khalil_state = branchrule.state
            sample_cand_scores = branchrule.obss_feats[0]['scores']
            sample_cands = np.where(sample_cand_scores != -1)[0]
            sample_cand_scores = sample_cand_scores[sample_cands]
            cand_choice = np.where(sample_cands == branchrule.targets[0])[0][0]

            root_filename = f"{outdir}/sample_root_0_{episode}.pkl"

            filenames.append(root_filename)
            with gzip.open(root_filename, 'wb') as f:
                pickle.dump({
                    'type':'root',
                    'episode':episode,
                    'instance': instance,
                    'seed': seed,
                    'stats': stats,
                    'root_state': [sample_state, sample_khalil_state, sample_cands, cand_choice, sample_cand_scores],
                    'obss': [branchrule.obss[0], branchrule.targets[0], branchrule.obss_feats[0], None],
                    'max_depth': max_depth
                    }, f)

            # node data
            for i in range(1, len(branchrule.obss)):
                iteration_counter = branchrule.obss_feats[i]['iteration']
                filenames.append(f"{outdir}/sample_node_{iteration_counter}_{episode}.pkl")
                with gzip.open(filenames[-1], 'wb') as f:
                    pickle.dump({
                        'type' : 'node',
                        'episode':episode,
                        'instance': instance,
                        'seed': seed,
                        'stats': stats,
                        'root_state': f"{outdir}/sample_root_0_{episode}.pkl",
                        'obss': [branchrule.obss[i], branchrule.targets[i], branchrule.obss_feats[i], None],
                        'max_depth': max_depth
                    }, f)

            out_queue.put({
                "type": "done",
                "episode": episode,
                "instance": instance,
                "seed": seed,
                "filenames":filenames,
                "nnodes":len(filenames),
            })

        m.freeProb()

def send_orders(orders_queue, instances, seed, time_limit, outdir, start_episode):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    orders_queue : multiprocessing.Queue
        Input queue from which orders are received.
    instances : list
        list of filepaths of instances which are solved by SCIP to collect data
    seed : int
        initial seed to insitalize random number generator with
    time_limit : int
        maximum time for which to solve an instance while collecting data
    outdir : str
        directory where to save data
    start_episode : int
        episode to resume data collection. It is used if the data collection process was stopped earlier for some reason.
    """
    rng = np.random.RandomState(seed)
    episode = 0
    while True:
        instance = rng.choice(instances)
        seed = rng.randint(2**32)
        # already processed; for a broken process; for root dataset to not repeat instances and seed
        if episode <= start_episode:
            episode += 1
            continue

        orders_queue.put([episode, instance, seed, time_limit, outdir, rng])
        episode += 1

def collect_samples(instances, outdir, rng, n_samples, n_jobs, time_limit):
    """
    Worker loop: fetch an instance, run an episode and record samples.

    Parameters
    ----------
    instances : list
        filepaths of instances which will be solved to collect data
    outdir : str
        directory where to save data
    rng : np.random.RandomState
        random number generator
    n_samples : int
        total number of samples to collect.
    n_jobs : int
        number of CPUs to utilize or number of instances to solve in parallel.
    time_limit : int
        maximum time for which to solve an instance while collecting data
    """
    os.makedirs(outdir, exist_ok=True)

    # start workers
    orders_queue = mp.Queue(maxsize=2*n_jobs)
    answers_queue = mp.SimpleQueue()
    workers = []
    for i in range(n_jobs):
        p = mp.Process(
                target=make_samples,
                args=(orders_queue, answers_queue),
                daemon=True)
        workers.append(p)
        p.start()

    # dir to keep samples temporarily; helps keep a prefect count
    tmp_samples_dir = f'{outdir}/tmp'
    os.makedirs(tmp_samples_dir, exist_ok=True)

    # if the process breaks due to some reason, resume from this last_episode.
    existing_samples = glob.glob(f"{outdir}/*.pkl")
    last_episode, last_i = -1, 0
    if existing_samples:
        last_episode = max(int(x.split("/")[-1].split(".pkl")[0].split("_")[-2]) for x in existing_samples) # episode is 2nd last
        last_i = len(existing_samples) # sample number is the last

    # start dispatcher
    dispatcher = mp.Process(
            target=send_orders,
            args=(orders_queue, instances, rng.randint(2**32), time_limit, tmp_samples_dir, last_episode),
            daemon=True)
    dispatcher.start()

    i = last_i # for a broken process
    in_buffer = 0
    while i <= n_samples:
        # print(f"begin sample {i}")
        sample = answers_queue.get()

        if sample['type'] == 'start':
            in_buffer += 1

        if sample['type'] == 'done':
            for filename in sample['filenames']:
                x = filename.split('/')[-1].split(".pkl")[0]
                os.rename(filename, f"{outdir}/{x}.pkl")
                i+=1
                print(f"[m {os.getpid()}] {i} / {n_samples} samples written, ep {sample['episode']} ({in_buffer} in buffer).")

                if  i == n_samples:
                    # early stop dispatcher (hard)
                    if dispatcher.is_alive():
                        dispatcher.terminate()
                        print(f"[m {os.getpid()}] dispatcher stopped...")
                    break

        if not dispatcher.is_alive():
            break

    # stop all workers (hard)
    for p in workers:
        p.terminate()

    shutil.rmtree(tmp_samples_dir, ignore_errors=True)