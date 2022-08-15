export FQDN=`grep -r "${HOSTNAME}" /etc/hosts | awk '{ print $1 }'`
export WORKSPACE_DIR=
export WORKSPACE_PATH=`pwd`/my_federation
export WORKSPACE_TEMPLATE=torch_histo

fx workspace create --prefix ${WORKSPACE_PATH} --template ${WORKSPACE_TEMPLATE}
cd ${WORKSPACE_PATH}
fx plan initialize
fx workspace certify
fx aggregator generate-cert-request --fqdn $FQDN
fx aggregator certify --fqdn $FQDN
fx workspace export







