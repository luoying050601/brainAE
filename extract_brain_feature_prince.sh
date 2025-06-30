#!/bin/bash
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
str=$"\n"
nohup /usr/bin/python3 -u alice_feature_extraction.py \
>> "prince_feature_extraction_$cur_date.out" 2>&1 &
sstr=$(echo -e $str)
echo $sstr