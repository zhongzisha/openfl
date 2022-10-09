split_num=$1
job_id=${SLURM_JOB_ID}
session_name=openfl_split${split_num}_${job_id}
tmux new-session -d -s ${session_name} bash
tmux send -t ${session_name}.0 "cd /data/zhongz2/openfl/; bash start_master_step2.sh;"
tmux send -t ${session_name}.0 c-m


