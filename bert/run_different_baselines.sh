export DEVICE=$1

for TASK_NAME in "rte"
do
    for SCALE in "0.0001" "0.005" "0.01" "0.05" "0.1" "0.2" 
    do 
        for POS in "7"
		do
            bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $DEVICE  >"./outlog/ablation/${TASK_NAME}-${SCALE}-${POS}.out" 2>&1
		done
    done
done

