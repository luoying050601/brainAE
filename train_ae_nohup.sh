#!/bin/bash

str=$"\n"
nohup /usr/bin/python3 -u /Storage/ying/project/brainAE/src/com/model/run_auto_encoder.py \
>> train_ae_9e-6.out 2>&1 &
sstr=$(echo -e $str)
echo $sstr