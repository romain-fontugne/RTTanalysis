import numpy as np
import dpcluster as dpc
# import pandas as pd


def loadData(filename):
    """Load a csv file in memory.
    :returns: pandas DataFrame with the file data

    """
    pass


def clusterRttPerIP(rttEstimates):

    # for each IP in the traffic
    ips = rttEstimates.ip.unique()

    for ip in ips:
        data = rttEstimates[rttEstimates.ip == ip].rtt
        vdp = dpgmm(data)
        params = NIWparam2Nparam(vdp)
        print(params)
        # write a file
        a = list()
        a.append("hello")


def NIWparam2Nparam(vdp, stdMax=100, minClusterSize=1):
    """
    Convert Gaussian Normal-Inverse-Wishart parameters to the usual Gaussian
    parameters (i.e. mean, standard deviation)
    """

    res = []
    mus, Sgs, k, nu = vdp.distr.prior.nat2usual(vdp.cluster_parameters()[vdp.cluster_sizes() > minClusterSize, :])[0]
    Sgs = Sgs / (k + 1 + 1)[:, np.newaxis, np.newaxis]
    for mu, Sg in zip(mus, Sgs):
        w, V = np.linalg.eig(Sg)
        V = np.array(np.matrix(V) * np.matrix(np.diag(np.sqrt(w))))
        if V < stdMax:
            res.append((mu, V))

    return res


def dpgmm(data, priorWeight=0.1, nbModes=16, thresh=1e-6, nbIter=100000):
    """
    Compute the Variational Inference for Dirichlet Process Mixtures
    on the given data.
    """
    if not len(data):
        return None
    data = np.array(data).reshape(-1, 1)
    vdp = dpc.VDP(dpc.distributions.GaussianNIW(1), w=priorWeight, k=nbModes, tol=thresh, max_iters=nbIter)
    vdp.batch_learn(vdp.distr.sufficient_stats(data))

    return vdp
