export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_PATH=`pwd`/my_federation
export WORKSPACE_TEMPLATE=torch_histo_mtl

fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
cd ${WORKSPACE_PATH}
fx plan initialize
fx workspace certify
fx aggregator generate-cert-request --fqdn $FQDN
fx aggregator certify --fqdn $FQDN --silent
fx workspace export



# on slaves
fx workspace import --archive WORKSPACE.zip


fx collaborator generate-cert-request -n {COL_LABEL}

# copy the col_*.zip to aggregator
fx collaborator certify --request-pkg /PATH/TO/col_{COL_LABEL}_to_agg_cert_request.zip --silent

# copy the agg_*.zip to the slave
fx collaborator certify --import /PATH/TO/agg_to_col_{COL_LABEL}_signed_cert.zip

fx aggregator start



fx collaborator start -n {COLLABORATOR_LABEL}




