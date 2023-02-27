export DEVICE=$1

for TASK_NAME in "rte" "cola" "mrpc" "stsb" 
do
    for SCALE in "0.0"
    do 
        for POS in "2" "7"
	do
		for LR in 2e-5 3e-5 4e-5 5e-5
		do
           		bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $LR $DEVICE  >"./outlog/tune-bert/${TASK_NAME}-${SCALE}-${POS}-${LR}.out" 2>&1
		done
	done
    done
done

