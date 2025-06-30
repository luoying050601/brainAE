#!/bin/bash
cur_date="$(date "+%Y-%m-%d-%H:%M:%S")"

str=$"\n"
nohup /usr/bin/python3 -u src/com/model/run_auto_encoder_prince.py \
>>"train_prince_ae_${cur_date}".out 2>&1 &
sstr=$(echo -e $str)
echo $sstr