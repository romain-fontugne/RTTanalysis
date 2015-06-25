# RTTanalysis
Estimation and analysis of round trip time from TCP traffic.

These scripts are written in python and traffic traces are read with Ipsumdump. Please make sure you have python and Ipsumdump (https://github.com/kohler/ipsumdump) installed on your computer. The analysis part requires certain python packages that are easily installed with "pip install":
* numpy
* pandas
* dpcluster
* matplotlib (optional)

## RTT Estimation
To compute all RTT estimates for each pair of packets (SEQ/ACK) from a pcap file, run the following command:
```Shell
ipsumdump -tsSdDFQKL --eth-dst -r pcapFile.dump.gz --filter="tcp" | python rttEstimation_TCP.py > rtt.csv
```

Ipsumdump can deal with either a raw or compressed pcap file.
If you have to deal with large pcap files, pypy can do the same thing in less time:
```Shell
ipsumdump -tsSdDFQKL --eth-dst -r pcapFile.dump.gz --filter="tcp" | pypy rttEstimation_TCP.py > rtt.csv
```

The resulting csv file look like this:
```Shell
head rtt.csv
209.114.162.19,134.222.11.65,0.001618,00-16-9C-7C-B0-00
209.114.168.106,217.134.46.150,0.000250,00-16-9C-7C-B0-00
209.114.162.19,134.222.11.65,0.001878,00-16-9C-7C-B0-00
209.114.168.106,97.241.163.189,0.000363,00-16-9C-7C-B0-00
209.114.163.88,33.143.242.160,0.000294,00-16-9C-7C-B0-00
209.114.169.91,49.196.160.103,0.005866,00-16-9C-7C-B0-00
209.114.183.62,119.180.210.20,0.008859,00-16-9C-7C-B0-00
168.203.147.150,71.29.55.34,0.009997,00-16-9C-7C-B0-00
119.180.210.20,209.114.183.62,0.000618,00-0E-39-E3-34-00
209.114.183.62,119.180.210.20,0.009848,00-16-9C-7C-B0-00
```

The first and third columns are the most important ones. The third column is the estimated RTT for the IP address given by the first column.
The second and fourth columns are the destination IP and MAC address of the TCP acknowledgment packet used to estimate the corresponding RTT.
Note that multiple RTT estimates are computed for the same TCP flow.

## RTT Analysis
The goal here is to identify groups of similar RTT values for each IP address. The dpgmm.py script does this by finding mixed RTT distributions using the Dirichlet Process Gaussian Mixture Model. This script takes as input the CSV file generated previously:
```Shell
python dpgmm.py rtt.csv outputDirectory/
```
And it creates for each IP address a CSV file that looks like this:
```Shell
cat outputDirectory/209.114.162.19.csv
0.00162277498992,3.3783390488e-06
0.00150017549559,3.77950997701e-06
0.00174587674708,4.1721750336e-06
0.00187348282214,4.16300079454e-06
0.00199608008361,4.96783666398e-06
```
Each line represents an RTT distribution and the two values are the corresponding mean and standard deviation.

# Author
Romain Fontugne (National Institute of Informatics) http://romain.fontugne.free.fr
