#!/bin/bash
set -euo pipefail
mkdir -p "./oe"
workDir=$(pwd)

conditions=(3A1 \
3A13L \
3A2 \
3A23L \
3B1 \
3B13L-d1 )


################################################################
## Long SA runs for each Networks trained on 6 conditions

#######################################
## Maximxzing
d=0.30
n_iter=20000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=2000000
init_per_condition=25

#############
## First 25
#for ((c_idx=3; c_idx<4; c_idx++ ))
#        do
#        cond=${conditions[$c_idx]}
#        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
#        p=$( ls ../../forAlex_6conditions/weights/$cond/*.hdf5 )
#        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | head -n $init_per_condition  > $inOut_idxs
#        fo=${cond}_max_d.${d}_samples.set1_iter.20M
#        name=SA_${cond}_max_d.${d}_iter.20M
#        sbatch -p gpusong --mem 200G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
##!/bin/bash
#source ~/lib/loadModules_TensorFlow.bash
#python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval
#EOF
#done

################
## Second 25
#for ((c_idx=3; c_idx<4; c_idx++ ))
#        do  
#        cond=${conditions[$c_idx]}
#        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
#        p=$( ls ../../forAlex_6conditions/weights/$cond/*.hdf5 )
#        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | tail -n $init_per_condition  > $inOut_idxs
#        fo=${cond}_max_d.${d}_samples.set2_iter.20M
#        name=SA_${cond}_max_d.${d}_iter.20M_set2
#        sbatch -p gpusong --mem 200G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
##!/bin/bash
#source ~/lib/loadModules_TensorFlow.bash
#python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval
#EOF
#done

## Longer runs for 3B1 and 3B13L-d1
d=0.30
n_iter=30000000
sampleInterval=2
n_save=20000
n_updates=2
saveInterval=2000000
init_per_condition=25

################
## First 25
#for ((c_idx=4; c_idx<5; c_idx++ ))
#        do
#        cond=${conditions[$c_idx]}
#        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
#        p=$( ls ../../forAlex_6conditions/weights/$cond/*.hdf5 )
#        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | head -n $init_per_condition  > $inOut_idxs
#        fo=${cond}_max_d.${d}_samples.set1_iter.30M
#        name=SA_${cond}_max_d.${d}_iter.30M_set1
#        sbatch -p gpusong --mem 200G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
##!/bin/bash
#source ~/lib/loadModules_TensorFlow.bash
#python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval
#EOF
#done

################
## Second 25
for ((c_idx=4; c_idx<6; c_idx++ ))
        do  
        cond=${conditions[$c_idx]}
        inOut_idxs=${cond}_idxs_highest${init_per_condition}.txt
        p=$( ls ../../forAlex_6conditions/weights/$cond/*.hdf5 )
        tail -n+2 ./initializations/${cond}_highest50.tsv | cut -f1 | tail -n $init_per_condition  > $inOut_idxs
        fo=${cond}_max_d.${d}_samples.set2_iter.30M
        name=SA_${cond}_max_d.${d}_iter.30M_set2
        sbatch -p gpusong --mem 200G --gres gpu:1 --nodelist=compute-3-2 -D $workDir --job-name=$name -o "./oe/$name.o" -e "./oe/$name.e" <<EOF
#!/bin/bash
source ~/lib/loadModules_TensorFlow.bash
python ../codes/SA.py  --inOut_idxs $inOut_idxs -p $p --n_iter $n_iter --n_save $n_save -d $d --n_updates $n_updates --sampleInterval $sampleInterval  --fo $fo --saveInterval $saveInterval
EOF
done


#######################################
## Minimizing (Need to Run this)
