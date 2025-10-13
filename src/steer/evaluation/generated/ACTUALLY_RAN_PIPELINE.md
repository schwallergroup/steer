 ```python
 # Generate features: extracted from routes' feasibility analysis
 python src/steer/evaluation/generated/extract_features_v2.py --max-workers 20 --routes-per-file 3

# Generate verifier codes for each of the features
 python src/steer/evaluation/generated/pipeline_v2.py     --input-dir data/outputs/2025-10-12_093739     --max-workers 100

# Register and run all the codes in benchmark, and compute std of scores (make sure they vary)
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

# Finally run generated benchmark with some model
python src/steer/evaluation/generated/run_with_cli.py     --benchmark src/steer/evaluation/generated/     --cache-path data/outputs/2025-10-12_093739     --model gemini-2.5-pro

# Results are here
grep "correlation" data/outputs/generated_benchmark_2025-10-12_211710/evaluation_results.json | awk '{if($2!="NaN," && $2>0.) print $2}' | wc -l

# Total 161 results, 154 are not nan, 119 are finite > 0.
 grep "correlation"  data/outputs/generated_benchmark_2025-10-12_211710/evaluation_results.json | awk '{sub(",", "", $2); if($2!="NaN") {a+=$2;c+=1}}END{print a/c}' 

# avg score: 0.2870

# Results with claude-3-5-sonnet:
grep "correlation"  data/outputs/generated_benchmark_2025-10-13_100121/evaluation_results.json | awk '{if($2!="NaN," && $2>0.) print $2}' | wc -l
# 157 non-nan

grep "correlation"  data/outputs/generated_benchmark_2025-10-13_100121/evaluation_results.json | awk '{sub(",", "", $2); if($2!="NaN") {a+=$2;c+=1}}END{print a/c}'
# avg score: 0.1904


```