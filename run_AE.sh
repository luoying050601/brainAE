#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
nohup python3.7 -u src/com/model/run_auto_encoder_shell.py \
--MODEL 'albert-xlarge-v2' \
>> "output/albert-xlarge-v2_$cur_date".out 2>&1 &