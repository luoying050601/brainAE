#horovodrun -np 8 \
# shellcheck disable=SC2006
cur_date="`date "+%Y-%m-%d-%H:%M:%S"`"
nohup python3.7 -u src/com/model/2VS2Test.py \
>> "output/2VS2Test_$cur_date".out 2>&1 &