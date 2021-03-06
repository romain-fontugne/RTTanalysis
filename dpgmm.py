import numpy as np
import glob
import dpcluster as dpc
import pandas as pd
import os
import sys
try:
    import matplotlib.pylab as plt
    import matplotlib as mpl
except Exception, e:
    sys.stderr.write("Matplotlib is not available!")
    


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
                usecols=["rtt","start_sec"])
            val = fi.split("/")
            tmp["ip"] = "{0}->{1}".format(val[-4], val[-3])
            data.append(tmp)

        df = pd.concat(data)

    # The ip addresses become the index
    df = df.set_index("ip")

    return df


def clusterRTToverTime(rttEstimates, timeBin="60", outputDirectory="./rttDistributions/",
        minEstimates=10, plot=True, logNormal=True):
    """For each IP address, find the different RTT distributions for each time 
    bin and plot the average value of each distribution.
    """

    # for each IP in the traffic
    ips = rttEstimates.index.unique()
    for ip in ips:
        start = rttEstimates[rttEstimates.index == ip].start_sec.min()
        end = rttEstimates[rttEstimates.index == ip].start_sec.max()
        dataIP = rttEstimates[rttEstimates.index == ip]

        x = []
        y = []
        z = []

        i = 0

        for ts in range(start,end,timeBin):
            if logNormal:
                data = np.log10(dataIP[(dataIP.start_sec>=ts) & (dataIP.start_sec<ts+timeBin)].rtt)
            else:
                data = dataIP[(dataIP.start_sec>=ts) & (dataIP.start_sec<ts+timeBin)].rtt
          
            # Look only at flows containing a certain number of RTT estimates
            if len(data) < minEstimates:
                sys.stderr("Ignoring data!! not enough samples!")
                continue
            
            # Cluster the data
            vdp = dpgmm(data)
            if vdp is None:
                continue

            params = NIWparam2Nparam(vdp)
            if logNormal:
                mean, std = logNormalMeanStdDev(params[0, :], params[1, :])
            else:
                mean = params[0, :]
                std = params[1, :]

            for mu, sig in zip(mean, std):
                y.append(mu) 
                z.append(sig)
                x.append(ts)

        # Plot the clusters characteristics in a file
        plt.figure()
        plt.errorbar(x,y,yerr=z,fmt="o")
        plt.grid(True)
        if logNormal:
            plt.savefig("{0}/{1}_timeBin{2}sec_logNormal.eps".format(outputDirectory, ip, timeBin))
        else:
            plt.savefig("{0}/{1}_timeBin{2}sec_normal.eps".format(outputDirectory, ip, timeBin))


def clusterRttPerIP(rttEstimates, outputDirectory="./rttDistributions/", minEstimates=10, plot=True, logNormal=False):
    """For each IP address, find the different RTT distributions and write
    their mean and standard deviation in files.
    """

    # for each IP in the traffic
    ips = rttEstimates.index.unique()

    for ip in ips:
        if logNormal:
            data = np.log10(rttEstimates[rttEstimates.index == ip].rtt)
        else:
            data = rttEstimates[rttEstimates.index == ip].rtt

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
        if logNormal:
            mean, std = logNormalMeanStdDev(params[0, :], params[1, :])
        else:
            mean = params[0, :]
            std = params[1, :]

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
    
    res = np.zeros( (len(mus), 2) )
    for i, (mu, Sg) in enumerate(zip(mus, Sgs)):
        w, V = np.linalg.eig(Sg)
        V = np.array(np.matrix(V) * np.matrix(np.diag(np.sqrt(w))))
        V = V[0]
        res[i] = (mu[0], V[0])

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


def dpgmm(data, priorWeight=0.1, maxClusters=32, thresh=1e-3, maxIter=10000):
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
    stats = vdp.distr.sufficient_stats(data)
    vdp.batch_learn(stats)

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
    minorLocator = mpl.ticker.MultipleLocator(10)
    ax.xaxis.set_minor_locator(minorLocator)
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

    if filename.endswith(".csv"):
        # Get RTT data from given file
        rtt = loadData(filename, format="rttEstimate")

        # Sample RTT estimates: samplingRate=0.1 means that 10% of the
        # estimates will be used
        samplingRate = 0.1
        if samplingRate:
            rtt = rtt.sample(frac=samplingRate)
        
        # Find RTT distributions for each IP address
        clusterRttPerIP(rtt, outputDirectory, logNormal=False)

    else:
        # Get RTT data from given file
        rtt = loadData(filename, format="thomas")
        # Find RTT distributions over time
        clusterRTToverTime(rtt, 600, outputDirectory, logNormal=False)
        #clusterRttPerIP(rtt, outputDirectory)
