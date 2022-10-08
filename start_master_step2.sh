
source ~/.bashrc
source ~/source_cuda102.sh
source /data/zhongz2/venv_py38_openfl/bin/activate

cd /lscratch/${SLURM_JOB_ID}
export FL_VERSION=v2
export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation_${FL_VERSION}
export WORKSPACE_TEMPLATE=torch_histo_mtl_${FL_VERSION}
cd my_federation_${FL_VERSION}

echo "start aggregator"
fx aggregator start




