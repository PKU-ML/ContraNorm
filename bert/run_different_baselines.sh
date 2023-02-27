export DEVICE=$1

for TASK_NAME in "qnli"
do
    for SCALE in "0.005" "0.01" "0.02" "0.04" "0.06" "0.08" "0.1" 
    do 
        for POS in "7"
	do
		for EPOCH in "5" 
		do
			for LAYER in "12"
        		do
            			bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $LAYER $EPOCH $DEVICE  >"./outlog/ablation/${TASK_NAME}-${SCALE}-${POS}-${LAYER}-epoch${EPOCH}.out" 2>&1
       			done
	       	done
	done
    done
done

