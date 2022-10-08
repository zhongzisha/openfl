
source ~/.bashrc
source ~/source_cuda102.sh
source /data/zhongz2/venv_py38_openfl/bin/activate

cd /lscratch/${SLURM_JOB_ID}
export FL_VERSION=v2
export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation_${FL_VERSION}
export WORKSPACE_TEMPLATE=torch_histo_mtl_${FL_VERSION}

############
split_num=$1
num_nodes=$2
node_index=$3
master_name=$4

echo "import the files"
fx workspace import --archive /data/zhongz2/temp/my_federation_${FL_VERSION}.zip
cd my_federation_${FL_VERSION}

echo "copy data to node"
if [ ! -d ./data/all ]; then
  python /data/zhongz2/openfl/openfl-workspace/torch_histo_mtl_${FL_VERSION}/copy_cache_files.py ${split_num} ${node_index}
fi

echo "fx collaborator generate-cert-request"
fx collaborator generate-cert-request -n ${node_index} -d ${node_index} --silent
cp col_${node_index}_to_agg_cert_request.zip /data/zhongz2/temp/
