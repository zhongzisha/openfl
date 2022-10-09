node_index=$1
split_num=$2
job_id=${SLURM_JOB_ID}
session_name=openfl_split${split_num}_${job_id}
tmux new-session -d -s ${session_name} bash
tmux send -t ${session_name}.0 "cd /data/zhongz2/openfl/; bash start_node_step3.sh ${node_index};"
tmux send -t ${session_name}.0 c-m


