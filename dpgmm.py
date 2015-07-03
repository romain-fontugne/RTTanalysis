import numpy as np
import glob
import dpcluster as dpc
import pandas as pd
import os
import sys
try:
    import matplotlib.pylab as plt
except Exception, e:
    sys.stderr("Matplotlib is not available!")
    


def loadData(filename, format="rttEstimate"):
    """Load a csv file in memory.
    :returns: pandas DataFrame with the file data

    """

    if format=="rttEstimate":
        df = pd.read_csv(filename, sep=",", header=None, names=["ip", "peer", "rtt", "dstMac"])
    elif format=="thomas":
        # the filename is a directory containing several RTT measurements
        # ..../ipSrc/ipDst/flowID/hour
        data = []
        for fi in glob.glob(filename):
            tmp = pd.read_csv(fi, sep="\t", comment="s", header=None,
                names=["rtt", "start_sec", "start_msec", "end_sec", "end_msec"],
                usecols=["rtt"])
            val = fi.split("/")
            tmp["ip"] = "{0}->{1}".format(val[-4], val[-3])
            data.append(tmp)

        df = pd.concat(data)

    # The ip addresses become the index
    df = df.set_index("ip")

    return df


def clusterRttPerIP(rttEstimates, outputDirectory="./rttDistributions/", minEstimates=10, plot=True):
    """For each IP address, find the different RTT distributions and write
    their mean and standard deviation in files.
    """

    # for each IP in the traffic
    ips = rttEstimates.index.unique()

    for ip in ips:
        data = np.log10(rttEstimates[rttEstimates.index == ip].rtt)

        # Look only at flows containing a certain number of RTT estimates
        if len(data) < minEstimates:
            continue
        
        # Cluster the data
        vdp = dpgmm(data)
        if vdp is None:
            continue

        # Write the clusters characteristics in a file
        fi = open("{0}/{1}.csv".format(outputDirectory, ip), "w")
        params = NIWparam2Nparam(vdp)
        mean, std = logNormalMeanStdDev(params[0, :], params[1, :])
        for mu, sig in zip(mean, std):
            fi.write("{0},{1}\n".format(mu, sig))

        if plot:
            plotRttDistribution(rttEstimates, ip, "{0}/{1}.eps".format(outputDirectory, ip))


def NIWparam2Nparam(vdp, minClusterIPRatio=0.05):
    """
    Convert Gaussian Normal-Inverse-Wishart parameters to the usual Gaussian
    parameters (i.e. mean, standard deviation)

    :vdp: Variational Dirichlet Process obtained from dpgmm
    :minClusterIPRatio: Ignore distributions standing for a ratio of IPs lower
    than minClusterIPRatio
    """

    nbIPs = float(np.sum(vdp.cluster_sizes()))
    mus, Sgs, k, nu = vdp.distr.prior.nat2usual(vdp.cluster_parameters()[
        vdp.cluster_sizes() > (minClusterIPRatio * nbIPs), :])[0]
    Sgs = Sgs / (k + 1 + 1)[:, np.newaxis, np.newaxis]
    
    res = np.zeros( (2, len(mus)) )
    for i, (mu, Sg) in enumerate(zip(mus, Sgs)):
        w, V = np.linalg.eig(Sg)
        V = np.array(np.matrix(V) * np.matrix(np.diag(np.sqrt(w))))
        V = V[0]
        res[:, i] = (mu[0], V[0])

    return res


def logNormalMeanStdDev(loc, scale):
    """Compute the mean and standard deviation from the location and scale
    parameter of a lognormal distribution.

    :loc: location parameter of a lognormal distribution
    :scale: scale parameter of a lognmormal distribution
    :return: (mean,stdDev) the mean and standard deviation of the distribution
    """

    mu = 10 ** (loc + ((scale ** 2) / 2.0))
    var = (10 ** (scale ** 2) -1) * 10 ** (2 * loc + scale ** 2)

    return mu, np.sqrt(var)


def dpgmm(data, priorWeight=0.1, maxClusters=16, thresh=1e-6, maxIter=10000):
    """
    Compute the Variational Inference for Dirichlet Process Mixtures
    on the given data.

    :data: 1D array containing the data to cluster
    :priorWeight:  likelihood-prior distribution pair governing clusters.
    :maxClusters: Maximum number of clusters
    :
    """
    
    data = np.array(data).reshape(-1, 1)
    vdp = dpc.VDP(dpc.distributions.GaussianNIW(1), w=priorWeight, k=maxClusters, tol=thresh, max_iters=maxIter)
    vdp.batch_learn(vdp.distr.sufficient_stats(data))

    return vdp


def plotRttDistribution(rttEstimates, ip, filename, nbBins=500, logscale=False):
    """Plot the RTT distribution of an IP address

    :rttEstimates: pandas DataFrame containing the RTT estimations
    :ip: IP address to plot
    :filename: Filename for the plot
    :nbBins: Number of bins in the histogram
    :logscale: Plot RTTs in logscale if set to True
    :returns: None

    """

    if logscale:
        data = np.log10(rttEstimates[rttEstimates.index == ip].rtt)
    else:
        data = rttEstimates[rttEstimates.index == ip].rtt

    h, b=np.histogram(data, nbBins, normed=True)
    plt.figure(1, figsize=(9, 3))
    plt.clf()
    ax = plt.subplot()
    x = b[:-1]
    ax.plot(x, h, "k")
    ax.grid(True)
    plt.title("%s (%s RTTs)" % (ip, len(data)))
    if logscale:
        plt.xlabel("log10(RTT)")
    else:
        plt.xlabel("RTT")
    plt.ylabel("pdf")
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage: python {0} rtt.csv [outputDirectory]".format(sys.argv[0]))

    filename = sys.argv[1]

    if len(sys.argv) > 2:
        outputDirectory = sys.argv[2]

        # Create the output directory if it doesn't exist
        if not os.path.exists(outputDirectory):
            os.mkdir(outputDirectory)

    # Get RTT data from given file
    if filename.endswith(".csv"):
        rtt = loadData(filename, format="rttEstimates")
    else:
        rtt = loadData(filename, format="thomas")

    # Find RTT distributions for each IP address
    clusterRttPerIP(rtt, outputDirectory)
