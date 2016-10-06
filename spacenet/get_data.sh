#!/bin/bash

OUTPUT=$1

aws s3api get-object --bucket spacenet-dataset --key manifest.txt --request-payer requester $OUTPUT/manifest.txt

for i in $(cat $OUTPUT/manifest.txt | grep 'processedData.*\..*') ; do \
  echo mkdir -p $(dirname $OUTPUT/${i/\.\//}) && \
  echo Getting $i && \
  echo aws s3api get-object --bucket spacenet-dataset --key ${i/\.\//} --request-payer requester \
    $OUTPUT/${i/\.\//} ;\
done
