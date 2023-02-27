export DEVICE=$1

for DATA in "cora" "citeseer"
do
    for SCALE in "0.0" "0.2" "0.5" "0.8"
    do 
        for MODE in "CN"
		do
			for EPOCH in "200" 
			do
				for HID in "32" 
        		do
					for LAYER in "2" "4" "8" "16" "32" "64"
					do
						bash run_baselines.sh $DATA $MODE $SCALE $HID $LAYER $EPOCH $DEVICE  >"./outlog/${DATA}-${MODE}-${HID}-${SCALE}-${LAYER}-epoch${EPOCH}.out" 2>&1
					done
       			done
	       	done
		done
    done
done

