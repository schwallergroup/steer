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

# From the following results (116), 80 scores are > 0 or non-nan


# Results:

# wandb: corr_target_001 0.98088
# wandb: corr_target_002 0.0625
# wandb: corr_target_003 -0.74744
# wandb: corr_target_009 0.99274
# wandb: corr_target_010 0.12503
# wandb: corr_target_013 0.4351
# wandb: corr_target_015 -0.80751
# wandb: corr_target_016 -0.38753
# wandb: corr_target_017 0.13289
# wandb: corr_target_018 0.97397
# wandb: corr_target_020 -0.23551
# wandb: corr_target_023 -0.76287
# wandb: corr_target_029 1.0
# wandb: corr_target_030 0.93712
# wandb: corr_target_031 0.88359
# wandb: corr_target_033 -0.91085
# wandb: corr_target_036 0.87819
# wandb: corr_target_039 -0.15834
# wandb: corr_target_040 nan
# wandb: corr_target_043 0.34491
# wandb: corr_target_044 0.41596
# wandb: corr_target_045 -0.47068
# wandb: corr_target_046 0.90455
# wandb: corr_target_050 0.36627
# wandb: corr_target_051 0.83754
# wandb: corr_target_052 0.42128
# wandb: corr_target_053 0.84293
# wandb: corr_target_054 -0.13278
# wandb: corr_target_055 nan
# wandb: corr_target_058 0.29496
# wandb: corr_target_059 nan
# wandb: corr_target_060 0.99798
# wandb: corr_target_063 0.68705
# wandb: corr_target_064 -0.2401
# wandb: corr_target_066 -0.40588
# wandb: corr_target_067 1.0
# wandb: corr_target_071 0.04206
# wandb: corr_target_073 1.0
# wandb: corr_target_074 nan
# wandb: corr_target_075 -0.53912
# wandb: corr_target_077 0.55899
# wandb: corr_target_079 -0.61237
# wandb: corr_target_080 0.35592
# wandb: corr_target_081 0.2838
# wandb: corr_target_083 -0.29194
# wandb: corr_target_084 -0.16474
# wandb: corr_target_085 0.48936
# wandb: corr_target_086 0.67978
# wandb: corr_target_090 0.08281
# wandb: corr_target_091 0.93027
# wandb: corr_target_092 0.21665
# wandb: corr_target_093 0.45614
# wandb: corr_target_094 0.2501
# wandb: corr_target_097 0.09677
# wandb: corr_target_099 0.41863
# wandb: corr_target_100 0.75295
# wandb: corr_target_101 0.63095
# wandb: corr_target_102 0.32825
# wandb: corr_target_103 0.09149
# wandb: corr_target_104 0.49678
# wandb: corr_target_105 0.09333
# wandb: corr_target_106 0.17447
# wandb: corr_target_108 0.98439
# wandb: corr_target_111 0.56607
# wandb: corr_target_112 0.97947
# wandb: corr_target_115 0.99115
# wandb: corr_target_118 0.73421
# wandb: corr_target_119 0.62164
# wandb: corr_target_122 0.51029
# wandb: corr_target_125 -0.42126
# wandb: corr_target_127 0.0913
# wandb: corr_target_129 0.40858
# wandb: corr_target_131 -0.91988
# wandb: corr_target_132 -0.54916
# wandb: corr_target_133 0.01904
# wandb: corr_target_135 0.18974
# wandb: corr_target_137 -0.97035
# wandb: corr_target_139 nan
# wandb: corr_target_140 0.1222
# wandb: corr_target_142 0.45044
# wandb: corr_target_145 -0.36072
# wandb: corr_target_146 0.57427
# wandb: corr_target_148 -0.1016
# wandb: corr_target_149 -0.69896
# wandb: corr_target_150 0.72136
# wandb: corr_target_151 -0.42717
# wandb: corr_target_154 0.96737
# wandb: corr_target_155 0.50491
# wandb: corr_target_156 0.21616
# wandb: corr_target_158 0.55301
# wandb: corr_target_159 0.76835
# wandb: corr_target_160 0.20826
# wandb: corr_target_161 0.09221
# wandb: corr_target_162 0.99989
# wandb: corr_target_163 0.69799
# wandb: corr_target_164 0.56478
# wandb: corr_target_165 -0.13897
# wandb: corr_target_167 0.03027
# wandb: corr_target_168 0.21085
# wandb: corr_target_169 0.15342
# wandb: corr_target_170 0.36863
# wandb: corr_target_171 0.3962
# wandb: corr_target_172 -0.79121
# wandb: corr_target_174 0.73762
# wandb: corr_target_175 -0.10509
# wandb: corr_target_176 nan
# wandb: corr_target_178 0.92372
# wandb: corr_target_180 0.245
# wandb: corr_target_181 -0.4827
# wandb: corr_target_182 0.39675
# wandb: corr_target_183 0.31565
# wandb: corr_target_184 nan
# wandb: corr_target_185 -0.21171
# wandb: corr_target_188 -0.97967
# wandb: corr_target_189 0.91984
# wandb: corr_target_190 0.4863
```