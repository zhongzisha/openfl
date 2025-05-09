export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation
export WORKSPACE_TEMPLATE=torch_histo_mtl

fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
cd ${WORKSPACE_PATH}
# copy data to data/all
fx plan initialize

fx workspace certify
fx aggregator generate-cert-request --fqdn $FQDN
fx aggregator certify --fqdn $FQDN --silent
fx workspace export



# on slave 1
export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
fx workspace import --archive my_federation.zip
cd my_federation
python /data/zhongz2/openfl/openfl-workspace/torch_histo_mtl/copy_cache_files.py 1 img

fx collaborator generate-cert-request -n {COL_LABEL}

# copy the col_*.zip to aggregator
fx collaborator certify --request-pkg /PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip --silent

# copy the agg_*.zip to the slave
fx collaborator certify --import /PATH/TO/agg_to_col_{COL_LABEL}_signed_cert.zip

fx collaborator start -n {COLLABORATOR_LABEL}

# on master
fx aggregator start


