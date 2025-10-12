 ```python
 python src/steer/evaluation/generated/extract_features_v2.py --max-workers 20 --routes-per-file 3
 python src/steer/evaluation/generated/pipeline_v2.py     --input-dir data/outputs/2025-10-12_093739     --max-workers 1
 python src/steer/evaluation/generated/pipeline_v2.py     --input-dir data/outputs/2025-10-12_093739     --max-workers 100


 python src/steer/evaluation/generated/run_evaluation.py     --benchmark src/steer/evaluation/generated/generated_prompt_specs_v2_sampled.json --codes-dir=src/steer/evaluation/generated/eval_types_v2
 
 # Filter tasks that are valid (i.e. no errors and std>0)
 python src/steer/evaluation/generated/filter_valid_tasks.py \
    --results src/steer/evaluation/generated/evaluation_results.json \
    --benchmark src/steer/evaluation/generated/generated_prompt_specs_v2_sampled.json \
    --output src/steer/evaluation/generated/prompt_specs_filtered.json
# results: 161 valid tasks generated (out of 557 originally)
# Filtered out:
#   - Errors: 46
#   - Zero std: 350
#   - NaN std: 0
 ```