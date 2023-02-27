export DEVICE=$1

for TASK_NAME in "rte" "stsb" "cola" "mrpc" "sst2"
do
    for SCALE in "0.001" "0.005" "0.01" "0.02" "0.05" "0.1" "0.2"
    do 
        for POS in "7"
	do
		for EPOCH in "5"
		do
			for LAYER in "12"
        		do
            			bash run_glue_baselines.sh $TASK_NAME $SCALE $POS $LAYER $EPOCH $DEVICE  >"./outlog/multistep/${TASK_NAME}-${SCALE}-${POS}-${LAYER}-epoch${EPOCH}-5step.out" 2>&1
       			done
	       	done
	done
    done
done

