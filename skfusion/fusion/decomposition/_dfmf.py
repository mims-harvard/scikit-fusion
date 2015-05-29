"""
============================================
Data Fusion by Matrix Factorization (`dfmf`)
============================================
"""

import logging
from operator import add
from collections import defaultdict
from functools import reduce

import numpy as np
import scipy.linalg as spla
from joblib import Parallel, delayed

from ._init import initialize


def __bdot(A, B, i, j, obj_types):
    entry = []
    if isinstance(list(A.values())[0], list):
        for l in range(len(A.get((i, j), []))):
            ll = [np.dot(A[i, k][l], B[k, j]) for k in obj_types
                  if (i, k) in A and (k, j) in B]
            if len(ll) > 0:
                tmp = reduce(add, ll)
                entry.append(np.nan_to_num(tmp))
    elif isinstance(list(B.values())[0], list):
        for l in range(len(B.get((i, j), []))):
            ll = [np.dot(A[i, k], B[k, j][l]) for k in obj_types
                  if (i, k) in A and (k, j) in B]
            if len(ll) > 0:
                tmp = reduce(add, ll)
                entry.append(np.nan_to_num(tmp))
    else:
        ll = [np.dot(A[i, k], B[k, j]) for k in obj_types
              if (i, k) in A and (k, j) in B]
        if len(ll) > 0:
            entry = reduce(add, ll)
            entry = np.nan_to_num(entry)
    return i, j, entry


def _par_bdot(A, B, obj_types, verbose, n_jobs):
    """Parallel block matrix multiplication.

    Parameters
    ----------
    A : dictionary of array-like objects
        Block matrix.

    B : dictionary of array-like objects
        Block matrix.

    obj_types : array-like
        Identifiers of object types.

    verbose : int
         The amount of verbosity.

    n_jobs: int (default=1)
        Number of jobs to run in parallel

    Returns
    -------
    C : dictionary of array-like objects
        Matrix product, A*B.
    """
    parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e3, verbose=verbose,
                            backend="multiprocessing")
    task_iter = (delayed(__bdot)(A, B, i, j, obj_types)
                 for i in obj_types for j in obj_types)
    entries = parallelizer(task_iter)
    C = {(i, j): entry for i, j, entry in entries if entry != []}
    return C


def _transpose(A):
    """Block matrix transpose.

    Parameters
    ----------
    A : dictionary of array-like objects
        Block matrix.

    Returns
    -------
    At : dictionary of array-like objects
        Block matrix with each of its block transposed.
    """
    At = {k: V.T for k, V in A.items()}
    return At


def count_objects(obj_types, R):
    """Count objects of each object type.

    Parameters
    ----------
    obj_types : set-like
        Object types

    R : dictionary of array-like objects
        Relation matrices

    Returns
    -------
    obj_type2n_obj : dictionary of int
        Number of objects per object type
    """
    obj_type2n_obj = {}
    for r in R:
        i, j = r
        for l in range(len(R[r])):
            for ax, obj_type in enumerate([i, j]):
                ni = obj_type2n_obj.get(obj_type, R[i,j][l].shape[ax])
                if ni != R[i,j][l].shape[ax]:
                    logging.critical("Relation matrix R_(%s,%s) dimension "
                                 "mismatch" % (i, j))
                obj_type2n_obj[obj_type] = ni

    if set(obj_types) != set(obj_type2n_obj.keys()):
        logging.critical("Object type specification mismatch")
    return obj_type2n_obj


def dfmf(R, Theta, obj_types, obj_type2rank,
         max_iter=10, init_type="random_vcol", stopping=None, stopping_system=None,
         verbose=0, compute_err=False, callback=None, random_state=None, n_jobs=1):
    """Data fusion by matrix factorization.

    Parameters
    ----------
    R : dictionary of array-like objects
        Relation matrices.

    Theta : dictionary of array-like objects
        Constraint matrices.

    obj_types : array-like
        Identifiers of object types.

    obj_type2rank : dict-like
        Factorization ranks of object types.

    max_iter : int, optional (default=10)
        Maximum number of iterations to perform.

    init_type : string, optional (default="random_vcol")
        The algorithm to initialize latent matrix factors.

    stopping : tuple (target_matrix, eps), optional (default=None)
        Stopping criterion. Terminate iteration if the reconstruction
        error of target matrix improves by less than eps.

    stopping_system : float, optional (default=None)
        Stopping criterion. Terminate iteration if the reconstruction
        error of the fused system improves by less than eps. compute_err is
        to True to compute the error of the fused system.

    verbose : int, optional (default=0)
         The amount of verbosity. Larger value indicates greater verbosity.

    compute_err : bool, optional (default=False)
        Compute the reconstruction error of every relation matrix if True.

    callback : callable, optional
        An optional user-supplied function to call after each iteration. Called
        as callback(G, S, cur_iter), where S and G are current latent estimates.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by np.random.

    n_jobs: int (default=1)
        Number of jobs to run in parallel

    Returns
    -------
    G :
    S :
    """
    verbose1 = verbose
    verbose = 50 - verbose
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p", level=verbose)

    obj_type2n_obj = count_objects(obj_types, R)
    R_to_init = {k: V[0] for k, V in R.items()}
    G = initialize(obj_types, obj_type2n_obj, obj_type2rank, R_to_init,
                   init_type, random_state)
    S = None

    if stopping:
        err = (None, None)
    if stopping_system:
        err_system = (None, None)
        compute_err = True

    logging.info("Solving for Theta_p and Theta_n")
    Theta_p, Theta_n = defaultdict(list), defaultdict(list)
    for r, thetas in Theta.items():
        for theta in thetas:
            t = theta > 0
            Theta_p[r].append(np.multiply(t, theta))
            Theta_n[r].append(np.multiply(t-1, theta))

    obj = []

    for iter in range(max_iter):
        if iter > 1 and stopping and err[1]-err[0] < stopping[1]:
            logging.info("Early stopping: target matrix change < %5.4f" \
                      % stopping[1])
            break
        if iter > 1 and stopping_system and \
                                err_system[1]-err_system[0] < stopping_system:
            logging.info("Early stopping: matrix system change < %5.4f" \
                      % stopping_system)
            break

        logging.info("Factorization iteration: %d" % iter)

        #######################################################################
        ########################### General Update ############################

        pGtG = {}
        for r in G:
            logging.info("Computing GrtGr: %s" % str(r))
            GrtGr = np.nan_to_num(np.dot(G[r].T, G[r]))
            pGtG[r] = spla.pinv(GrtGr)

        logging.info("Start to update S")

        tmp1 = _par_bdot(G, pGtG, obj_types, verbose1, n_jobs)
        tmp2 = _par_bdot(R, tmp1, obj_types, verbose1, n_jobs)
        tmp3 = _par_bdot(_transpose(G), tmp2, obj_types, verbose1, n_jobs)
        S = _par_bdot(pGtG, tmp3, obj_types, verbose1, n_jobs)

        #######################################################################
        ########################### General Update ############################

        logging.info("Start to update G")

        G_enum = {r: np.zeros(Gr.shape) for r, Gr in G.items()}
        G_denom = {r: np.zeros(Gr.shape) for r, Gr in G.items()}

        for r in R:
            i, j = r
            for l in range(len(R[r])):
                logging.info("Update G due to R_%s,%s^(%d)" % (i, j, l))

                tmp1 = np.dot(R[i, j][l], np.dot(G[j, j], S[i, j][l].T))
                tmp1 = np.nan_to_num(tmp1)
                t = tmp1 > 0
                tmp1p = np.multiply(t, tmp1)
                tmp1n = np.multiply(t-1, tmp1)

                tmp2 = np.dot(S[i, j][l], np.dot(G[j, j].T, np.dot(G[j, j], S[i, j][l].T)))
                tmp2 = np.nan_to_num(tmp2)
                t = tmp2 > 0
                tmp2p = np.multiply(t, tmp2)
                tmp2n = np.multiply(t-1, tmp2)

                tmp4 = np.dot(R[i, j][l].T, np.dot(G[i, i], S[i, j][l]))
                tmp4 = np.nan_to_num(tmp4)
                t = tmp4 > 0
                tmp4p = np.multiply(t, tmp4)
                tmp4n = np.multiply(t-1, tmp4)

                tmp5 = np.dot(S[i, j][l].T, np.dot(G[i, i].T, np.dot(G[i, i], S[i, j][l])))
                tmp5 = np.nan_to_num(tmp5)
                t = tmp5 > 0
                tmp5p = np.multiply(t, tmp5)
                tmp5n = np.multiply(t-1, tmp5)

                G_enum[i, i] += tmp1p + np.dot(G[i, i], tmp2n)
                G_denom[i, i] += tmp1n + np.dot(G[i, i], tmp2p)

                G_enum[j, j] += tmp4p + np.dot(G[j, j], tmp5n)
                G_denom[j, j] += tmp4n + np.dot(G[j, j], tmp5p)

        logging.info("Update of G due to constraint matrices")
        for r, thetas_p in Theta_p.items():
            logging.info("Considering Theta pos. %s" % str(r))
            for theta_p in thetas_p:
                G_denom[r] += np.dot(theta_p, G[r])
        for r, thetas_n in Theta_n.items():
            logging.info("Considering Theta neg. %s" % str(r))
            for theta_n in thetas_n:
                G_enum[r] += np.dot(theta_n, G[r])

        for r in G:
            G[r] = np.multiply(G[r], np.sqrt(
                np.divide(G_enum[r], np.maximum(G_denom[r], np.finfo(np.float).eps))))

        #######################################################################
        ######################## Reconstruction Error #########################

        if stopping:
            target, eps = stopping
            err = (np.linalg.norm(R[target]-np.dot(G[target[0],target[0]],
                    np.dot(S[target], G[target[1],target[1]].T))), err[0])

        if compute_err:
            s = 0
            for r in R:
                i, j = r
                for l in range(len(R[r])):
                    Rij_app = np.dot(G[i, i], np.dot(S[i, j][l], G[j, j].T))
                    r_err = np.linalg.norm(R[r][l]-Rij_app, "fro")
                    logging.info("Relation R_%s,%s^(%d) norm difference: " \
                              "%5.4f" % (i, j, l, r_err))
                    s += r_err
            logging.info("Error (objective function value): %5.4f" % s)
            obj.append(s)
            if stopping_system:
                err_system = (s, err_system[0])

        if callback:
            callback(G, S, iter)

    if compute_err:
        logging.info("Violations of optimization objective: %d/%d " % (
            int(np.sum(np.diff(obj) > 0)), len(obj)))
    return G, S


def transform(R_ij, Theta_i, target_obj_type, obj_type2rank, G, S,
              max_iter=10, init_type="random_c",
              stopping=None, stopping_system=None, verbose=0,
              compute_err=False, callback=None, random_state=None):
    verbose = 50 - verbose
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S %p", level=verbose)
    if isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        random_state = np.random.RandomState(random_state)

    n_targets = [R_ij[i,j][0].shape[0 if target_obj_type == i else 1] for i, j in R_ij]
    if len(set(n_targets)) > 1:
        logging.critical("Target object type: %s size mismatch" % target_obj_type)
    n_targets = n_targets[0]
    R_to_init = {k: V[0] for k, V in R_ij.items()}
    Gx = initialize(
        [target_obj_type], {target_obj_type: n_targets}, obj_type2rank,
        R_to_init, init_type, random_state)
    G_i = Gx[target_obj_type, target_obj_type]

    if stopping:
        err = (None, None)
    if stopping_system:
        err_system = (None, None)
        compute_err = True

    Theta_p, Theta_n = [], []
    for r, thetas in Theta_i.items():
        for theta in thetas:
            t = theta > 0
            Theta_p.append(np.multiply(t, theta))
            Theta_n.append(np.multiply(t-1, theta))

    obj = []

    for iter in range(max_iter):
        if iter > 1 and stopping and abs(err[1]-err[0]) < stopping[1]:
            logging.info("Early stopping: target matrix change < %5.4f" \
                      % stopping[1])
            break
        if iter > 1 and stopping_system and \
                                err_system[1]-err_system[0] < stopping_system:
            logging.info("Early stopping: matrix system change < %5.4f" \
                      % stopping_system)
            break

        logging.info("Factorization iteration: %d" % iter)

        #######################################################################
        ########################### General Update ############################

        logging.info("Start to update G")

        G_enum = np.zeros(G_i.shape)
        G_denom = np.zeros(G_i.shape)

        for r in R_ij:
            i, j = r
            for l in range(len(R_ij[r])):
                logging.info("Update G due to R_%s,%s^(%d)" % (i, j, l))
                if i is target_obj_type:

                    tmp1 = np.dot(R_ij[i, j][l], np.dot(G[j, j], S[i, j][l].T))
                    t = tmp1 > 0
                    tmp1p = np.multiply(t, tmp1)
                    tmp1n = np.multiply(t-1, tmp1)

                    tmp2 = np.dot(S[i, j][l], np.dot(G[j, j].T, np.dot(G[j, j], S[i, j][l].T)))
                    t = tmp2 > 0
                    tmp2p = np.multiply(t, tmp2)
                    tmp2n = np.multiply(t-1, tmp2)

                    G_enum += tmp1p + np.dot(G_i, tmp2n)
                    G_denom += tmp1n + np.dot(G_i, tmp2p)

                if j is target_obj_type:
                    tmp4 = np.dot(R_ij[i, j][l].T, np.dot(G[i, i], S[i, j][l]))
                    t = tmp4 > 0
                    tmp4p = np.multiply(t, tmp4)
                    tmp4n = np.multiply(t-1, tmp4)

                    tmp5 = np.dot(S[i, j][l].T, np.dot(G[i, i].T, np.dot(G[i, i], S[i, j][l])))
                    t = tmp5 > 0
                    tmp5p = np.multiply(t, tmp5)
                    tmp5n = np.multiply(t-1, tmp5)

                    G_enum += tmp4p + np.dot(G_i, tmp5n)
                    G_denom += tmp4n + np.dot(G_i, tmp5p)

        logging.info("Update of G due to constraint matrices")
        for theta_p in Theta_p:
            G_denom += np.dot(theta_p, G_i)
        for theta_n in Theta_n:
            G_enum += np.dot(theta_n, G_i)

        G_i = np.multiply(G_i, np.sqrt(
            np.divide(G_enum, np.maximum(G_denom, np.finfo(np.float).eps))))

        #######################################################################
        ######################## Reconstruction Error #########################

        if compute_err:
            s = 0
            for r in R_ij:
                i, j = r
                for l in range(len(R_ij[r])):
                    if i is target_obj_type:
                        Rij_app = np.dot(G_i, np.dot(S[i, j][l], G[j, j].T))
                        r_err = np.linalg.norm(R_ij[r][l]-Rij_app, "fro")
                    if j is target_obj_type:
                        Rij_app = np.dot(G[i, i], np.dot(S[i, j][l], G_i.T))
                        r_err = np.linalg.norm(R_ij[r][l]-Rij_app, "fro")
                    logging.info("Relation R_%s,%s^(%d) norm difference: " \
                          "%5.4f" % (i, j, l, r_err))
                    s += r_err
            logging.info("Error (objective function value): %5.4f" % s)
            obj.append(s)
            if stopping_system:
                err_system = (s, err_system[0])

        if callback:
            callback(G_i, iter)

    if compute_err:
        logging.info("Violations of optimization objective: %d/%d " % (
            int(np.sum(np.diff(obj) > 0)), len(obj)))
    return G_i


if 0:
    # toy example for factorizing new examples
    obj_types = [0, 1, 2]
    n1, n2, n3 = 400, 80, 264
    c1, c2, c3 = 30, 40, 50
    obj_type2rank = {0: c1, 1: c2, 2: c3}

    rnds = np.random.RandomState(0)

    R12 = 5 * rnds.rand(n1, n2)
    R13 = 20 * rnds.rand(n1, n3)
    R21 = R12.T
    R23 = 20 * rnds.rand(n2, n3)

    Theta = {}
    R = {(0, 1): [R12], (0, 2): [R13], (1, 2): [R23], (1, 0): [R21]}
    print("Inference")
    G, S = dfmf(
        R, Theta, obj_types, obj_type2rank, verbose=0, max_iter=10,
        init_type="random", compute_err=True, random_state=rnds)

    # new data
    o_n2_z3 = 20 * rnds.rand(3, n3)
    o_n2_z1 = 5 * rnds.rand(3, n1)

    print("We got some new data!")
    x_G2 = transform(
        {(1,0): [o_n2_z1], (0,1): [o_n2_z1.T], (1,2): [o_n2_z3]}, {},
        1, obj_type2rank, G, S, max_iter=10, init_type="random",
        compute_err=True, stopping=None, random_state=rnds)

    R12_hat = np.dot(G[0,0], np.dot(S[0,1][0], G[1,1].T))
    R21_hat = np.dot(G[1,1], np.dot(S[1,0][0], G[0,0].T))
    R23_hat = np.dot(G[1,1], np.dot(S[1,2][0], G[2,2].T))

    x_R12_hat = np.dot(G[0,0], np.dot(S[0,1][0], x_G2.T))
    x_R21_hat = np.dot(x_G2, np.dot(S[1,0][0], G[0,0].T))
    x_R23_hat = np.dot(x_G2, np.dot(S[1,2][0], G[2,2].T))

    err = np.linalg.norm(R23[0, :] - R23_hat[0, :])
    print("Error in R23 of object that existed before: %5.3f" % err)
    x_err = np.linalg.norm((o_n2_z3 - x_R23_hat)[0, :])
    print("Error in R23 of newly added object: %5.3f" % x_err)

    err = np.linalg.norm(R12[:, -1] - R12_hat[:, -1])
    print("Error in R12 of object that existed before: %5.3f" % err)
    x_err = np.linalg.norm((x_R12_hat - o_n2_z1.T)[:, 0])
    print("Error in R12 of newly added object: %5.3f" % x_err)

    err = np.linalg.norm(R21[0, :] - R21_hat[0, :])
    print("Error in R21 of object that existed before: %5.3f" % err)
    x_err = np.linalg.norm((x_R21_hat - o_n2_z1)[0, :])
    print("Error in R21 of newly added object: %5.3f" % x_err)
