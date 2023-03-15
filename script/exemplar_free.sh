
 

approach_name=ewc   #choices (finetuning  ewc lwf r_walk)
gpu_id=0

SEED=0
 
python -u src_incremental/main_incremental.py --exp-name base_${SEED} \
        --datasets bologna --num-tasks 4  --save-models --network fusion_fc --seed $SEED \
        --nepochs 200  --batch-size 128 --nc-first-task 5 --num-workers 8 --results-path   ./${approach_name}_exp_seed_${SEED} \
        --approach $approach_name --gpu $gpu_id --optimizer_type adam  --input residual_hist    
 