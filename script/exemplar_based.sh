
 

approach_name=r_walk   #choices (finetuning bic icarl il2m ewc lwf r_walk)
gpu_id=0

SEED=0
 
python -u src_incremental/main_incremental.py --exp-name base_${SEED} \
        --datasets bologna --num-tasks 4  --save-models --network fusion_fc --seed $SEED \
        --nepochs 200   --num-exemplars 500 --exemplar-selection random  --batch-size 128 --nc-first-task 5 --num-workers 8 --results-path   ./${approach_name}E_random_exp_500_seed_${SEED} \
        --approach $approach_name --gpu $gpu_id --optimizer_type adam  --input residual_hist    
 