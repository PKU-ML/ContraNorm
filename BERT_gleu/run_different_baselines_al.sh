export DEVICE=$1

for TASK_NAME in "rte" "cola" "mrpc" "stsb" "sst2"  
do
    for SCALE in "0.0001" "0.005" "0.01" "0.05" "0.1" "0.2"  
    do 
        for POS in "1" "2" "3" "4" "5" "6" "7"
		do
			bash run_glue_baselines_al.sh $TASK_NAME $SCALE $POS $DEVICE  >"./outlog/albert-base-v2/${TASK_NAME}-${SCALE}-${POS}.out" 2>&1
		done
    done
done

