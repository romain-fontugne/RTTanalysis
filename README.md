# RTTanalysis
Estimation and analysis of round trip time from TCP traffic.

These scripts are written in python and traffic traces are read with Ipsumdump. Please make sure you have python and Ipsumdump (https://github.com/kohler/ipsumdump) installed on your computer.

## Estimate RTTs from pcap/tcpdump files
To compute all RTT estimates for each pair of packets (SEQ/ACK) from a pcap file, run the following command:
```Shell
ipsumdump -tsSdDFQKL --eth-dst -r pcapFile.dump.gz --filter="tcp" | python rttEstimation_TCP.py > rtt.csv
```

Ipsumdump can deal with either a raw or compressed pcap file.
If you have to deal with large pcap files, pypy can do the same thing in less time:
```Shell
ipsumdump -tsSdDFQKL --eth-dst -r pcapFile.dump.gz --filter="tcp" | pypy rttEstimation_TCP.py > rtt.csv
```

### Output
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

# Author
Romain Fontugne (National Institute of Informatics) http://romain.fontugne.free.fr
