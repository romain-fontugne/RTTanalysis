#!/usr/bin/env python

import sys
from socket import inet_ntoa
from struct import unpack_from 
from collections import defaultdict


def rttEstimation(filename=None,binary=False):

  seq = {} 

  if binary:
    if filename!=None:
      fIn = open(filename,"rb")
    else:
      fIn = sys.stdin

    # Skip the header
    txt = fIn.readline()
    while not txt.startswith("!binary"):
      txt = fIn.readline()  
  
    #First Record length
    nextRec = fIn.read(4)

  else:
    if filename!=None:
      fIn = open(filename,"r")
    else:
      fIn = sys.stdin

    # Skip the header
    txt = fIn.readline()
    while not txt.startswith("!data"):
      txt = fIn.readline()

    txt = fIn.readline()

  #Read the trace
  while (binary and nextRec != "" and len(nextRec) == 4) or (not binary and txt):

    try:
    #For each packet get the timestamps, IP addresses, and TCP header fields
      if binary:
        # Get IPs, ports, size and protocol
        recLen  = unpack_from("!i",nextRec)[0]
        ts      = unpack_from('!II',fIn.read(8))
        #tsu     = unpack('!I',fIn.read(4))[0]    
        ipSrc   = inet_ntoa(fIn.read(4))
        portSrc = unpack_from('!H',fIn.read(2))[0]
        ipDst   = inet_ntoa(fIn.read(4))
        portDst = unpack_from('!H',fIn.read(2))[0]
        tcpFlags= unpack_from("!b",fIn.read(1))[0]
        tcpSeq  = unpack_from("!I",fIn.read(4))[0]
        tcpAck  = unpack_from("!I",fIn.read(4))[0]
        payload = unpack_from("!I",fIn.read(4))[0]
        ethDst = unpack_from('!BBBBBB',fIn.read(6)) # old versions of IPsumdump are doing wrong things here...
      else:
        word = txt.split()
        tsStr = word[0].split(".")
        ts = [int(tsStr[0]),int(tsStr[1])]
        ipSrc = word[1]
        portSrc = word[2]
        ipDst = word[3]
        portDst = word[4]
        tcpFlags = 0
        if "A" in word[5]:
          tcpFlags = tcpFlags | 16
        if "S" in word[5]:
          tcpFlags = tcpFlags | 2
        tcpSeq = int(word[6])
        tcpAck = int(word[7])
        payload= int(word[8])
        ethDst = word[9]
    except ValueError:
      #sys.stderr.write("WARN: ignored packet,"+txt+"\n")
      if binary:
        #next record length
        nextRec = fIn.read(4)
      else:
        txt =fIn.readline()
      continue
    
    if binary:
      #next record length
      nextRec = fIn.read(4)
    else:
      txt =fIn.readline()

    if tcpFlags & 2:           # Syn flag consume 1 byte
      tcpSeq += 1
    else:
      tcpSeq += payload

    # Hash keys to identify flows
    src = "{0}:{1}".format(ipSrc,portSrc)
    dst = "{0}:{1}".format(ipDst,portDst)
    
    if not src in seq:
      seq[src]={dst:{tcpSeq:ts}}
    elif not dst in seq[src]:
      seq[src][dst]={tcpSeq:ts}
    elif not tcpSeq in seq[src][dst]:
      seq[src][dst][tcpSeq] = ts
    else:                           # Retransmission
      seq[src][dst][tcpSeq] = None  # Reset the timer
    
    if tcpFlags & 16:          # ACK packet
      if dst in seq and src in seq[dst] and tcpAck in seq[dst][src]:
        ts0 = seq[dst][src][tcpAck]
        if ts0 != None:
          print("{0},{1},{2:f},{3}".format(ipSrc,ipDst,ts[0]-ts0[0]+(ts[1]-ts0[1])*1e-6,ethDst) )
          seq[dst][src][tcpAck] = None

if __name__ == "__main__":
  rttEstimation()
