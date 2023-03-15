 
SEED=0
gpu_id=0 


python amerini_comparison/main.py --results_path  ./amerini_result_${SEED}  --batch_size 128  --gpu $gpu_id  --num_workers 8 --seed $SEED  --net amerini --residual_type amerini 

 