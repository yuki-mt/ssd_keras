#!/bin/bash
set -e

options=$@
cd `dirname $0`/..

if [[ "$options" == *"local"* ]]; then
  docker_option="--local"
else
  docker_option=""
fi

sh/docker.sh $docker_option

python sagemaker/run.py $options -m `sh/get_image_name.sh`
