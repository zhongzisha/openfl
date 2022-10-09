
source ~/.bashrc
source ~/source_cuda102.sh
source /data/zhongz2/venv_py38_openfl/bin/activate

cd /lscratch/${SLURM_JOB_ID}
export FL_VERSION=v2
export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation_${FL_VERSION}
export WORKSPACE_TEMPLATE=torch_histo_mtl_${FL_VERSION}

split_num=$1
num_nodes=$2

echo "replace the split_num"
python /data/zhongz2/openfl/openfl-workspace/torch_histo_mtl_${FL_VERSION}/set_split_num.py ${split_num} ${num_nodes}
sleep 1

echo "fx workspace create"
fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
cd my_federation_${FL_VERSION}

echo "copy data"
python /data/zhongz2/openfl/openfl-workspace/torch_histo_mtl_${FL_VERSION}/gen_csv_for_nodes.py ${split_num} ${num_nodes}
mkdir -p data/all;
for f in `ls /data/zhongz2/tcga_brca_256/patch_images_cache_part1_0_np2048/all/*.txt`; do
  ln -sf $f data/all/;
done

echo "initialize on master"
fx plan initialize
fx workspace certify
fx aggregator generate-cert-request --fqdn $FQDN
fx aggregator certify --fqdn $FQDN --silent
fx workspace export

cp my_federation_${FL_VERSION}.zip /data/zhongz2/temp/;





