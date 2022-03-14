#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
nohup python3 -u src/com/model/run_auto_encoder.py \
>> "output/vanilla_ae_$cur_date".out 2>&1 &