node_index=$1
tmux new-session -d -s openfl bash
tmux send -t openfl.0 "cd /data/zhongz2/openfl/; bash start_node_step3.sh ${node_index};"
tmux send -t openfl.0 c-m


