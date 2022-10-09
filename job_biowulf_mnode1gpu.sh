#!/bin/bash

#SBATCH --job-name debug_openfl
#SBATCH --partition=gpu
#SBATCH --mail-user=ziszhong2022.slurm@gmail.com
#SBATCH --mail-type=ALL
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:p100:1,lscratch:512
##SBATCH --constraint=gpuk80
#SBATCH --cpus-per-task=16
#SBATCH --mem=90g
#SBATCH --ntasks-per-core=1
#SBATCH --time=240:00:00

#  sleep 100000000000000000000


echo ${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

source ~/.bashrc
source ~/source_cuda102.sh
source /data/zhongz2/venv_py38_openfl/bin/activate

split_num=4

nodenames_str=`python3 -c "import hostlist,os;print(','.join(hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])));"`
echo $nodenames_str
IFS=',' read -r -a nodenames <<< "$nodenames_str"
for index in "${!nodenames[@]}"
do
    echo "$index ${nodenames[index]}"
done

num_nodes=${#nodenames[@]}
let num_nodes-=1  # remove the master node

echo "run script on master"
index=0
master_name=${nodenames[index]}
ssh $master_name "cd /data/zhongz2/openfl/; bash start_master_step1.sh ${split_num} ${num_nodes}"

# in slaves
echo "run script on slaves step1"
for index in "${!nodenames[@]}"; do
  if [ $index -eq 0 ]; then continue; fi
  echo "$index ${nodenames[index]}"
  node_name=${nodenames[index]}
  ssh $node_name "cd /data/zhongz2/openfl/; bash start_node_step1.sh ${split_num} ${num_nodes} ${index} ${master_name}"
done
echo "run script on slaves step2"
for index in "${!nodenames[@]}"; do
  if [ $index -eq 0 ]; then continue; fi
  echo "$index ${nodenames[index]}"
  node_name=${nodenames[index]}
  ssh $master_name "cd /data/zhongz2/openfl/; bash start_node_step2.sh ${index}"
done
echo "run script on slaves step3"
for index in "${!nodenames[@]}"; do
  if [ $index -eq 0 ]; then continue; fi
  echo "$index ${nodenames[index]}"
  node_name=${nodenames[index]}
  ssh $node_name "cd /data/zhongz2/openfl/; bash start_node_step3_main.sh ${index} ${split_num};"
done

ssh $master_name "cd /data/zhongz2/openfl/; bash start_master_step2_main.sh ${split_num}"

echo "job is running"


sleep 10000000000000000

