source ~/.bashrc
source ~/source_cuda102.sh
source /data/zhongz2/venv_py38_openfl/bin/activate

cd /lscratch/${SLURM_JOB_ID}
export FL_VERSION=v2
export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation_${FL_VERSION}
export WORKSPACE_TEMPLATE=torch_histo_mtl_${FL_VERSION}
cd my_federation_${FL_VERSION}

node_index=$1
echo "fx collaborator certify"
fx collaborator certify --import /data/zhongz2/temp/agg_to_col_${node_index}_signed_cert.zip
echo "start collaborator ${node_index}"
unset http_proxy https_proxy
fx collaborator start -n ${node_index}


