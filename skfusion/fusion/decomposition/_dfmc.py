"""
=========================================
Data Fusion by Matrix completion (`dfmc`)
=========================================
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


def _update_G_for_Rij(Rij_l, G_i, G_j, Sij_l):
    """Multiplicative update of latent factors G_i and G_j
    due to l-th relation matrix R_ij.

    Parameters
    ----------
    Rij_l :
        The l-th relation matrix for R_ij

    G_i :  array-like, shape (n_objects_i, n_latent_i)
        Current estimate of latent factor G_i

    G_j : array-like, shape (n_objects_j, n_latent_j)
        Current estimate of latent factor G_j

    Sij_l : array-like, shape(n_latent_i, n_latent_j)
        Current estimation of the l-th latent factor S_ij

    Returns
    -------
    G_enum_i :
    G_denom_i :
    G_enum_j :
    G_denom_j :
    """
    tmp1 = np.dot(Rij_l, np.dot(G_j, Sij_l.T))
    t = tmp1 > 0
    tmp1p = np.multiply(t, tmp1)
    tmp1n = np.multiply(t-1, tmp1)

    tmp2 = np.dot(Sij_l, np.dot(G_j.T, np.dot(G_j, Sij_l.T)))
    t = tmp2 > 0
    tmp2p = np.multiply(t, tmp2)
    tmp2n = np.multiply(t-1, tmp2)

    tmp4 = np.dot(Rij_l.T, np.dot(G_i, Sij_l))
    t = tmp4 > 0
    tmp4p = np.multiply(t, tmp4)
    tmp4n = np.multiply(t-1, tmp4)

    tmp5 = np.dot(Sij_l.T, np.dot(G_i.T, np.dot(G_i, Sij_l)))
    t = tmp5 > 0
    tmp5p = np.multiply(t, tmp5)
    tmp5n = np.multiply(t-1, tmp5)

    G_enum_i = tmp1p + np.dot(G_i, tmp2n)
    G_denom_i = tmp1n + np.dot(G_i, tmp2p)

    G_enum_j = tmp4p + np.dot(G_j, tmp5n)
    G_denom_j = tmp4n + np.dot(G_j, tmp5p)

    return (G_enum_i, G_denom_i), (G_enum_j, G_denom_j)


def dfmc(R, M, Theta, obj_types, obj_type2rank, max_iter=10,
         init_type="random_vcol", stopping=None, stopping_system=None,
         verbose=0, compute_err=False, callback=None, random_state=None,
         n_jobs=1):
    """Data fusion by matrix completion.

    Parameters
    ----------
    R : dictionary of array-like objects
        Relation matrices.

    M : dictionary of array-like objects
        Mask matrices

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
    R_to_init = {k: V[0] for k, V in R.items()}
    obj_type2n_obj = count_objects(obj_types, R)
    G = initialize(obj_types, obj_type2n_obj, obj_type2rank, R_to_init,
                   init_type, random_state)
    S = None

    if stopping:
        err_target = (None, None)
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

    # For reasons of matrix completion the data should be copied inside DFMC
    R = {r: [m.copy() for m in data] for r, data in R.items()}

    obj = []
    for iter in range(max_iter):
        if iter > 1 and stopping and err_target[1]-err_target[0] < stopping[1]:
            logging.info("Early stopping: target matrix change < %5.4f" \
                      % stopping[1])
            break
        if iter > 1 and stopping_system and \
                                err_system[1]-err_system[0] < stopping_system:
            logging.info("Early stopping: matrix system change < %5.4f" \
                      % stopping_system)
            break

        logging.info("Completion iteration: %d" % iter)

        #######################################################################
        ########################## Matrix Completion ##########################

        if iter == 0:
            for r in M:
                for l in range(len(R[r])):
                    if M[r][l] is None:
                        continue
                    R[r][l][M[r][l]] = 0.

        #######################################################################
        ########################### General Update ############################

        pGtG = {}
        for r in G:
            logging.info("Computing GrtGr: %s" % str(r))
            GrtGr = np.nan_to_num(np.dot(G[r].T, G[r]))
            # numpy.linalg.pinv approximates the Moore-Penrose pseudo inverse
            # using SVD (the lapack method dgesdd), whereas scipy.linalg.pinv
            # solves a linear system in the least squares sense (the lapack
            # method dgelss). Alternatively, numpy.linalg.solve solve a
            # linear system in the least squares sense. The latter turns out
            # to be unstable in this case.
            pGtG[r] = spla.pinv(GrtGr)

        logging.info("Start to update S")

        tmp1 = _par_bdot(G, pGtG, obj_types, verbose1, n_jobs)
        tmp2 = _par_bdot(R, tmp1, obj_types, verbose1, n_jobs)
        tmp3 = _par_bdot(_transpose(G), tmp2, obj_types, verbose1, n_jobs)
        S = _par_bdot(pGtG, tmp3, obj_types, verbose1, n_jobs)

        #######################################################################
        ########################## Matrix Completion ##########################

        for r in M:
            for l in range(len(M[r])):
                if M[r][l] is None:
                    continue
                i, j = r
                Rij_app = np.dot(G[i, i], np.dot(S[i, j][l], G[j, j].T))
                R[r][l][M[r][l]] = Rij_app[M[r][l]]

        #######################################################################
        ########################### General Update ############################

        logging.info("Start to update G")

        G_enum = {r: np.zeros(Gr.shape) for r, Gr in G.items()}
        G_denom = {r: np.zeros(Gr.shape) for r, Gr in G.items()}

        for r in R:
            i, j = r
            logging.info("Update G due to R_%s,%s" % (i, j))
            # Start parallel update
            logging.info("Start parallel update of R_%s,%s" % (i, j))

            parallelizer = Parallel(n_jobs=n_jobs, max_nbytes=1e3,
                                    backend="multiprocessing", verbose=verbose1)
            task_iter = (delayed(_update_G_for_Rij)(
                    R[i,j][l], G[i,i], G[j,j], S[i,j][l]) for l in range(len(R[r])))
            G_ij_update = parallelizer(task_iter)

            for (G_enum_i, G_denom_i), (G_enum_j, G_denom_j) in G_ij_update:
                G_enum[i, i] += G_enum_i
                G_denom[i, i] += G_denom_i

                G_enum[j, j] += G_enum_j
                G_denom[j, j] += G_denom_j

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

        if stopping:
            target, eps = stopping
            r, l = target
            err_target = (np.linalg.norm(R[r][l]-np.dot(G[r[0],r[0]],
                    np.dot(S[r][l], G[r[1],r[1]].T))), err_target[0])

        if compute_err:
            s = 0
            for r in R:
                i, j = r
                for l in range(len(R[r])):
                    Rij_app = np.dot(G[i, i], np.dot(S[r][l], G[j, j].T))
                    r_err = np.linalg.norm(R[r][l]-Rij_app, "fro")
                    logging.info("Relation %s^(%d) norm difference: " \
                              "%5.4f" % (str(r), l, r_err))
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


if 0:
    # toy example for matrix completion
    rnds = np.random.RandomState(0)
    from sklearn import datasets

    R12 = datasets.load_digits(7).data
    obj_types = [0, 1, 2]
    obj_type2rank = {0: 100, 1: 200, 2:10}

    R12b = np.random.rand(R12.shape[0], R12.shape[1])
    R12b[R12b < 0.3] = 0

    # mask R12
    grid = np.indices(R12.shape)
    idx = list(zip(grid[0].ravel(), grid[1].ravel()))
    rnds.shuffle(idx)
    idxi, idxj = list(zip(*idx[:int(0.9 * R12.size)]))
    mR12 = np.ma.array(R12)
    mR12[idxi, idxj] = np.ma.masked
    print("R12 mask: %d" % mR12.mask.sum())

    R23 = np.random.rand(R12.shape[1], 40)

    R = {(0,1): [R12, R12b], (1,2): [R23.copy(), R23]}
    Theta = {}
    M = {(0,1): [mR12.mask, mR12.mask.copy()]}

    ## target matrix
    i, j = 0, 1
    ##

    # Evaluate completion (mask)
    G, S = dfmc(R, M, Theta, obj_types, obj_type2rank,
                max_iter=10, init_type="random_vcol", stopping=None,
                stopping_system=None, verbose=0, compute_err=True,
                callback=None, random_state=rnds, n_jobs=1)

    Rij_app = np.dot(G[i,i], np.dot(S[i, j][0], G[j, j].T))
    X_pred = R[i,j][0][M[i,j][0]] - Rij_app[M[i,j][0]]
    oob_fro = np.sqrt(np.sum(X_pred**2) / X_pred.size)
    print("OOB(Completion) RMSE for R(%d, %d): %5.3f" % (i+1, j+1, oob_fro))

    rnd = R[i,j][0][M[i,j][0]]
    rnds.shuffle(rnd)
    X_rnd = R[i,j][0][M[i,j][0]] - rnd
    rnd_fro = np.sqrt(np.sum(X_rnd**2) / X_rnd.size)
    print("Rnd RMSE for R(%d, %d): %5.3f" % (i+1, j+1, rnd_fro))


    print("R12 set to zero: %d" % np.size(R12[idxi, idxj]))

    O = {(0,1): [R12.copy(), R12b.copy()], (1,2): [R23.copy()]}
    M = {}
    R12[idxi, idxj] = 0
    R = {(0,1): [R12, R12b], (1,2): [R23]}

    def cb(G, S, cur_iter):
        print('Iteration: %d' % cur_iter)

    # Evaluate factorization (no mask)
    G, S = dfmc(R, M, Theta, obj_types, obj_type2rank,
                max_iter=10, init_type="random_vcol", stopping=None,
                stopping_system=None, verbose=0, compute_err=False,
                callback=cb, random_state=rnds, n_jobs=1)

    Rij_app = np.dot(G[i,i], np.dot(S[i, j][0], G[j, j].T))
    X_pred = O[i,j][0][idxi, idxj] - Rij_app[idxi, idxj]
    oob_fro = np.sqrt(np.sum(X_pred**2) / X_pred.size)
    print("OOB(Factorization) RMSE for R(%d, %d): %5.3f" % (i+1, j+1, oob_fro))
