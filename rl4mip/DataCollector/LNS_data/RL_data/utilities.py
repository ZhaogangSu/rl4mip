import numpy as np
import pyscipopt as scip
import torch

def init_scip_params(model, seed, heuristics=True, presolving=True, separating=False, conflict=True):

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

def init_scip_paramsH(model, seed, heuristics=False, presolving=True, separating=False, conflict=True):

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

def extract_state(model, buffer=None):
    """
    Compute a bipartite graph representation of the solver. In this
    representation, the variables and constraints of the MILP are the
    left- and right-hand side nodes, and an edge links two nodes iff the
    variable is involved in the constraint. Both the nodes and edges carry
    features.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    buffer : dict
        A buffer to avoid re-extracting redundant information from the solver
        each time.
    Returns
    -------
    variable_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the variable nodes in the bipartite graph.
    edge_features : dictionary of type ('names': list, 'indices': np.ndarray, 'values': np.ndarray}
        The features associated with the edges in the bipartite graph.
        This is given as a sparse matrix in COO format.
    constraint_features : dictionary of type {'names': list, 'values': np.ndarray}
        The features associated with the constraint nodes in the bipartite graph.
    """
    if buffer is None or model.getNNodes() == 1:

        buffer = {}

    # update state from buffer if any
    s = model.getBipartiteGraphRepresentation(buffer['scip_state'] if 'scip_state' in buffer else None)
    buffer['scip_state'] = s

    ncons = len(model.getConss())
    nvars = len(model.getVars())

    variable_features = {}
    variable_features['names'] = list(s[3]['col'].keys())
    variable_features['values'] = np.array(s[0])

    constraint_features = {}
    constraint_features['names'] = list(s[3]['row'].keys())
    constraint_features['values'] = np.array(s[2])

    edge_features = {}
    edge_features['names'] = ['coef_normalized']
    edge_features['indices'] = np.array([
    [row[1] for row in s[1]],
    [row[0] for row in s[1]]
    ])

    edge_features['values'] = np.array([row[2] for row in s[1]])

    values = torch.tensor(edge_features['values'].squeeze(), dtype=torch.float32)  # shape: (250000,)

    indices = torch.tensor(edge_features['indices'], dtype=torch.long)  # shape: (2, 250000)

    A = torch.sparse_coo_tensor(indices, values, (ncons, nvars))
    # A = torch.sparse_coo_tensor(edge_features['indices'], edge_features['values'], (ncons, nvars))
        # c_nodes.append()
    dense_matrix = A.to_dense().numpy()
    edge_features['incidence'] = dense_matrix
    
    return constraint_features, variable_features, indices, edge_features, 