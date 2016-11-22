#!/bin/bash

if [ "$(uname)" == "Darwin" ]; then
  UNZIP="tar -xf"
else
  UNZIP="unzip -nq"
fi

WORK_DIR=$(pwd)
ANNOTATION_DIR="annotation"
BASE_URL="http://press.liacs.nl/mirflickr/mirflickr25k"
IMAGE="mirflickr25k.zip"
ANNOTATION="mirflickr25k_annotations_v080.zip"

# Download annotations
mkdir -p "${ANNOTATION_DIR}"
cd ${ANNOTATION_DIR}
echo "Downloading ${ANNOTATION} to $(pwd)"
wget -nd -c "${BASE_URL}/${ANNOTATION}"
echo "Unzipping ${ANNOTATION}"
${UNZIP} ${ANNOTATION}
rm "README.txt"
rm ${ANNOTATION}

cd ${WORK_DIR}

#Dowload images
echo "Downloading ${IMAGE} to $(pwd)"
wget -nd -c "${BASE_URL}/${IMAGE}"
echo "Unzipping ${IMAGE}"
${UNZIP} ${IMAGE}
rm ${IMAGE}

#Create output json
echo "Running create_json_annotation.py script"
python create_json_annotation.py
