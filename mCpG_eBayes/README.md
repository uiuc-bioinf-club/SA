# Files:
`XX.posterior.tsv` -> parameters and posterior mean for beta distribtuion at each CpG site with coverage >0 (considering both samples) on chromosomes 1-4
	- Columns:
	 	- seq — chromosome
		- idx — genomic index (zero based ?)
		- strand — 
		- nMe — number observed methylated (across 2 replicates)
		- nUMe — number observed unmethylated (across 2 replicates)
		- total — nME + nUMe
		- .alpha1 — alpha parameter of posterior Beta distrib
		- .beta1  — beta parameter of posterior Beta distrib
		- .fitted — posterior mean
		- .raw — MLE est of p 
		- .low  — some lower bound on baysian credible interval (ignore)
		- .high — some upper bound on baysian credible interval (ignore)
`XX.prior.txt` -> Parameters of prior beta distribution 

# Method for posterior estimation of p mCpG using empirical bayes
for each condition: 3A1, 3A2, 3A13L , 3A23L, 3B1, 3B13L-d1
	1. read counts of mCpG  and umCpG from Bismark outputs for both samples
	2. Restrict to CpG's on chromosomes 1-4 and remove CpG's with no coverage in either condition
	3. Fit beta prior parameters by empirical bayes and write these to fule : `eBB_XXX.prior.txt`
	4. Write parameters and posterior mean estimates to file `eBB_XXX.posterior.tsv`
	5. visualize in `./compare_mCpG_byCondition.ipynb` and `./checkBetaFit.ipynb` 
