
tmux new-session -d -s openfl bash
tmux send -t openfl.0 "cd /data/zhongz2/openfl/; bash start_master_step2.sh ${index};"
tmux send -t openfl.0 c-m


