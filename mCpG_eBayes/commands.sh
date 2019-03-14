#!/bin/bash
set -euo pipefail

##############################################################################
## Compute paramters of beta prior for each conditions.
## Compute parameter of beta posterior for each CpG give condition prior 
## - These commands were run on AF2_laptop
BSseqOutdir="/Users/afinneg2/projects/DNAMethylation/processedData/\
BSseq_replicates/progressiveAlign4"  #TODO: update
conditions=("3A1" "3A2" "3B1" "3A13L" "3A23L" "3B13L-d1")
sampleDirs=( BSseq_3A_sample1_align,BSseq_3A_sample2_align \
BSseq_3A2_sample1_align,BSseq_3A2_sample2_align \
BSseq_3B1_sample1_align,BSseq_3B1_sample2_align \
BSseq_3A3L_sample1_align,BSseq_3A3L_sample2_align \
BSseq_3A23L_sample1_align,BSseq_3A23L_sample2_align \
BSseq_growthNb1_d1_align,BSseq_growthNb2_d1_align )  ## TODO: update

for (( i=0;i<${#conditions[@]}; i++ ))
	do
	condition=${conditions[$i]}
        files=($( echo ${sampleDirs[$i]} | tr ',' ' '))
	for f in ${files[@]}
		do
		f_full=$BSseqOutdir/$f/CpG_CytosineReport.txt
		cat $f_full | sort -k1,1 -k2,2n > ${f}.sorted.tsv
	done
	fo=dME_bayes_eBB_$condition
	echo --f1 ${files[0]}.sorted.tsv --f2 ${files[1]}.sorted.tsv --fo $fo
	Rscript ./fitbetaBinom_ebbr.R --f1 ${files[0]}.sorted.tsv --f2 ${files[1]}.sorted.tsv --fo $fo
	rm ${files[0]}.sorted.tsv ${files[1]}.sorted.tsv 
done

###########################################################################
## Re-sort the XX.posterior.tsv files by chromosome and then position. Rename

for c in ${conditions[@]}
	do
	head -n1 dME_bayes_eBB_${c}.posterior.tsv > header.txt
	tail -n+2 dME_bayes_eBB_${c}.posterior.tsv | sort -k1,1 -k2,2n > ${c}.posterior.tmp.tsv 
	cat header.txt ${c}.posterior.tmp.tsv > ${c}.posterior.tsv 
	mv dME_bayes_eBB_${c}.prior.txt ${c}.prior.txt
	rm dME_bayes_eBB_${c}.posterior.tsv  ${c}.posterior.tmp.tsv 
done
rm header.txt	




