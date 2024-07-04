#!/bin/bash
source config.sh

#az group create -l $LOCATION -n $RESOURCE_GROUP

#az acr create --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --sku Standard --admin-enabled true

az appservice plan create -g $RESOURCE_GROUP -n $SERVICE_PLAN_NAME -l $LOCATION --is-linux --sku P1V3

#az acr build --registry $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --image $IMAGE_NAME .

az webapp create -g $RESOURCE_GROUP -p $SERVICE_PLAN_NAME -n $WEBAPP_NAME -i $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest


