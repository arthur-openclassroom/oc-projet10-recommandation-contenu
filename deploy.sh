#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

RG=rg-mycontent-p10
LOC=germanywestcentral

[ -f artifacts/snapshot.npz ] || .venv/bin/python scripts/prepare_artifacts.py

az account show -o none 2>/dev/null || az login -o none
SUFFIX=$(az account show --query id -o tsv | cut -c1-6)
ST=stmycontent$SUFFIX
FUNC=func-mycontent-$SUFFIX

for ns in Microsoft.Storage Microsoft.Web Microsoft.Insights Microsoft.OperationalInsights; do
    az provider register --namespace $ns --wait
done
az group create --name $RG --location $LOC -o none
az storage account create --name $ST --resource-group $RG --location $LOC --sku Standard_LRS -o none
az functionapp show --name $FUNC --resource-group $RG -o none 2>/dev/null ||
    az functionapp create --name $FUNC --resource-group $RG --storage-account $ST \
        --flexconsumption-location $LOC --runtime python --runtime-version 3.11 \
        --instance-memory 512 -o none

az storage container create --account-name $ST --name artifacts --auth-mode key -o none
az storage blob upload --account-name $ST --container-name artifacts --name snapshot.npz \
    --file artifacts/snapshot.npz --overwrite --auth-mode key -o none

(cd azure_function && func azure functionapp publish $FUNC --python)

URL=https://$(az functionapp show --name $FUNC --resource-group $RG --query defaultHostName -o tsv)/api/recommend
KEY=$(az functionapp keys list --name $FUNC --resource-group $RG --query functionKeys.default -o tsv)
umask 077
printf 'FUNCTION_URL=%s\nFUNCTION_KEY=%s\n' "$URL" "$KEY" > .env