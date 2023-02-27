export DEVICE=$1

for TASK_NAME in "rte" "cola" "mrpc" "stsb" "sst2"  
do
    for SCALE in "0.0001" "0.005" "0.01" "0.05" "0.1" "0.2"  
    do 
        for POS in "1" "2" "3" "4" "5" "6" "7" "8"
	do
		for EPOCH in "5" 
		do
			for LAYER in "12"
        		do
            			bash run_glue_baselines_al.sh $TASK_NAME $SCALE $POS $LAYER $EPOCH $DEVICE  >"./outlog/albert-base-v2/new/${TASK_NAME}-${SCALE}-${POS}-${LAYER}-epoch${EPOCH}.out" 2>&1
       			done
	       	done
	done
    done
done

