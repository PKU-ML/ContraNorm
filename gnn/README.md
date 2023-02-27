# ContraNorm 
Official pytorch source code for ContraNorm [paper](https://openreview.net/pdf?id=SM7XkJouWHm) (ICLR 2023)  

## usage
For GCN with ContraNorm, we tune the scale in {0.2, 0.5, 0.8, 1.0}. Empirically, a higher scale is preferred in a deeper layer setting. 

For usage, you can type the following command in the terminal for different settings.
```
python main.py --data cora --model DeepGCN --nlayer 16 --norm_mode CN --norm_scale 1.0 --use_layer_norm --hid 32 --epochs 200
```
You can also use the script to tune the hyperparameters.
```
bash run_different_baselines.sh 0
```
Here, the single number denotes the index of GPU.

## citation
If you use our code, please cite
    ```

    ```
