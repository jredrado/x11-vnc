#!/bin/bash
source config.sh

#az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $WEBAPP_NAME --settings @settings.json

# re-tag the image and push it to the repository
docker build -t tst .

az acr login -n $CONTAINER_REGISTRY

docker image tag tst $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest
docker push $CONTAINER_REGISTRY.azurecr.io/$IMAGE_NAME:latest

#az webapp restart --name $WEBAPP_NAME --resource-group $RESOURCE_GROUP
