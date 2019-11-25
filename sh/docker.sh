#!/bin/bash

set -e

local_option=$1
is_local="--local"
if [ -n "$local_option" -a "$local_option" != $is_local ]
then
    echo 'argument is nothing or "--local"' >&2
    exit 1
fi


cd `dirname $0`/..

image_name=`sh/get_image_name.sh`
docker build -t $image_name -f docker/Dockerfile .

if [ "$local_option" != $is_local ]
then
  $(aws ecr get-login --no-include-email)
  docker push $image_name
fi
