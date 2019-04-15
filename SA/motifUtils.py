#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:25:03 2019

@author: afinneg2
"""
from __future__ import division
from __future__ import print_function
import numpy as np
from collections import OrderedDict
from scipy.stats import  entropy

import matplotlib.pyplot as plt
import  viz_seq

plt.rcParams["axes.titlesize"] = "large"
plt.rcParams["axes.labelsize"] = "large"
plt.rcParams["xtick.labelsize"] = "large"
plt.rcParams["ytick.labelsize"] = "large"
plt.rcParams["legend.fontsize"] = "large"

###############################################################################################################
### PLOTTING 
def KLdiv_fromSamples(samples, q = np.array(4*[0.25]) ):
    """
    samples - (nSamples ,4, seqLen). each slice samples[i, : , k] is a one hot encoding of sequence
    """
    if len(samples.shape) == 2:
        ## interpret samples as a probability distribution over single nucleotides
        if samples.shape[0] !=4:
            raise Exception("samples must have shape (nSamples ,4, seqLen) or (4, seqLen)")
        p = samples
    else:
        if samples.shape[1] != 4:
            raise Exception("samples must have shape (nSamples ,4, seqLen)")
        p = samples.mean(axis = 0)
    q = np.repeat(q[:, None] ,  repeats = samples.shape[-1], axis= 1 )
    KL_div = np.array( [ entropy(p[:,i], q[:,i]) for  i in range(q.shape[-1])  ] )
    return KL_div
    


def plot_sNTdistrib(samples, ax, motif= True, q = np.array(4*[0.25]) ,
                    kws_plotWeights ={ "height_padding_factor": 0.2,
                                        "length_padding": 1.0 ,
                                        "subticks_frequency": 10.0 , 
                                        "highlight": {} } ):
    """
    samples - (nSamples ,4, seqLen). each slice samples[i, : , k] is a one hot encoding of sequence
             Axis 1 should have one hot encoding of nuceotides with basis0 -> A, basis1 -> C, ""->G , ""->T
    """
    def plot_seqArrContinuous(arr , ax = None, 
                          colorDict = OrderedDict([('A', 'green'), ('C', 'blue' ) , 
                                                   ( "G", "orange") , ('T' ,'red' )]),
                         figsize = (7.5,3), legendLoc= 0, ):
        """
        Inputs 
        --------
            arr = (4, seqLen) where axis 0 is A, C, G, T
            ax  - matplotlib ax or None
        """
        if ax is None: 
            fig, ax = plt.subplots(figsize =figsize)
        xVals = np.arange(0 , arr.shape[1])
        for i , nt in  enumerate(colorDict.keys()):
            ax.plot( xVals ,arr[i , :], color = colorDict[nt], label = nt, marker = 'o',
                   ms = 2, alpha = 0.7)
        ax.legend(loc = legendLoc )
        ax.set_xlabel("Input Position")
        ax.set_ylabel("Score")
        return ax
        
    if len(samples.shape) == 2:
            ## interpret samples as a probability distribution over single nucleotides
            if samples.shape[0] !=4:
                raise Exception("samples must have shape (nSamples ,4, seqLen) or (4, seqLen)")
            freq = samples
    else:
        if samples.shape[1] != 4:
                raise Exception("samples must have shape (nSamples ,4, seqLen)")
        freq = samples.mean(axis = 0) 
    if motif:
        KL_div = KLdiv_fromSamples(freq , q = q )
        viz_seq.plot_weights_given_ax(ax = ax, array = freq*KL_div ,
                                      **kws_plotWeights)
        #ax = plt.gca()
        ax.set_ylabel("KL divergence")
    else:
        ax = plot_seqArrContinuous(freq, ax = ax)
        ax.set_ylabel("Score")
    return ax

########################################################################################################################
### WRITE MOTIFS
    
def writeMotif_memeFormat( freqMat, motifName="test"  ):
    """
    feqMat - (seqLen ,4)
    """
    header=\
"""MEME version 4.4

ALPHABET= ACGT

strands: + -

Background letter frequencies:
A 0.25 C 0.25 G 0.25 T 0.25

"""
    header+="MOTIF {} {}\n\n".format(motifName, motifName)
    if freqMat.shape[1] != 4:
        if freqMat.shape[0] ==4:
            freqMat = freqMat.transpose()
        else:
            raise Exception( "feqMat.shape must be (seqLen ,4)")
    
    header+="letter-probability matrix: alength= 4 w= {}\n".format(freqMat.shape[0])
    motifData = ""
    for ntFreq in freqMat:
        motifData+="{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(*ntFreq)
    return header + motifData[:-1]
    
    
