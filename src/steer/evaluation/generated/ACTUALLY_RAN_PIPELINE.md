 ```python
 python src/steer/evaluation/generated/extract_features_v2.py --max-workers 20 --routes-per-file 3
 python src/steer/evaluation/generated/pipeline_v2.py     --input-dir data/outputs/2025-10-12_093739     --max-workers 1
 python src/steer/evaluation/generated/pipeline_v2.py     --input-dir data/outputs/2025-10-12_093739     --max-workers 100
 python src/steer/evaluation/generated/run_evaluation.py     --benchmark src/steer/evaluation/generated/generated_prompt_specs_v2_sampled.json --codes-dir=src/steer/evaluation/generated/eval_types_v2
 ```