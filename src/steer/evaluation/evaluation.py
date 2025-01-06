"""Run evaluation script."""

# Benchmark stats
import json
from eval_types import *
from steer.evaluation import load_default_tasks
from steer.logger import setup_logger

logger = setup_logger()

RESULTS_PATH = '../../../../synthegy/steer/'
PROMPT_TYPE = 'fullroute_no_feasibility'

def get_latest_file(path, fid):
    import os
    files = os.listdir(path)
    files = [f for f in files if fid in f]
    files.sort()
    if files:
        return f'{path}/{files[-1]}'
    else:
        return None

def run_task(task):
    filename = get_latest_file(f'{RESULTS_PATH}/{PROMPT_TYPE}', task.id)
    if filename is None:
        print(f"File not found for {task.id}")
        return None

    with open(filename, 'r') as f:
        data = json.load(f)

    gt_score, lmscore = task.evaluate(data)
    return gt_score, lmscore


tasks = load_default_tasks(".")

for task in tasks:
    result = run_task(task)
    if result is not None:
        logger.info(f"{task.id}: {result[0]}, {result[1]}")
