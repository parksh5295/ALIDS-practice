#!/bin/bash

datasets=('WUSTLIIoT' 'ToNIoT' 'EdgeIIoT' 'XIIoTID')
# datasets=('XIIoTID' 'ToNIoT')
options=('multi' 'binary')
# options=('binary')
samplers=('TPESampler' 'CmaEsSampler' 'NSGAIISampler' 'QMCSampler')
# samplers=('CmaEsSampler' 'NSGAIISampler')

for option in "${options[@]}"; do
    for dataset in "${datasets[@]}"; do
        for sampler in "${samplers[@]}"; do

            echo "$option classification of $dataset - $sampler"
            python3 main.py --data "$dataset" --option "$option" --sampler "$sampler"

            # if [[ "$dataset" == "XIIoTID" && "$sampler" == "NSGAIISampler" && "$option" == "binary" ]]; then
            #     echo "$option classification of $dataset - $sampler"
            #     python3 main.py --data "$dataset" --option "$option" --sampler "$sampler"
            # fi
            # if [[ "$dataset" == "XIIoTID" && "$sampler" == "QMCSampler" && "$option" == "binary" ]]; then
            #     echo "$option classification of $dataset - $sampler"
            #     python3 main.py --data "$dataset" --option "$option" --sampler "$sampler"
            # fi
            # if [[ "$dataset" == "ToNIoT" && "$option" == "binary" ]]; then
            #     echo "$option classification of $dataset - $sampler"
            #     python3 main.py --data "$dataset" --option "$option" --sampler "$sampler"
            # fi 

            
        done
    done
done

echo "Results saved"