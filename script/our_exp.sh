 
SEED=0
gpu_id=0 


python amerini_comparison/main.py --results_path  ./our_result_${SEED}  --batch_size 128  --gpu $gpu_id  --num_workers 8 --seed $SEED  --net our_fusion --residual_type our 

 