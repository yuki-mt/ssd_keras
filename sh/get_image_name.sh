#!/bin/bash

set -e

region='ap-northeast-1'
image_name='ssd'

account_name=`aws sts get-caller-identity | jq -r '.Account'`
ecr_uri="$account_name.dkr.ecr.$region.amazonaws.com"

echo "$ecr_uri/$image_name:`git rev-parse --short HEAD`"
