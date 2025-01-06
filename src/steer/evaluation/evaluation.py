"""Run evaluation script."""

# Benchmark stats
import json
from .eval_types import *
from steer.evaluation import _load_default_tasks

RESULTS_PATH = '../../../../synthegy/steer/'
PROMPT_TYPE = 'fullroute'


# task # defined in benchmark: smiles, prompt, eval_class
# Class needs to define eval method (e.g. what class to use for eval, and internally target_depth)

tasks = _load_default_tasks()

for task in tasks:
    with open(f'{RESULTS_PATH}/{PROMPT_TYPE}/{task.id}.json', 'r') as f:
        data = json.load(f)

    evaluator = task.eval_class()
    evaluator.eval(data, task)





def route_score(x):
    # Disconnection happens (!=-1), + should happen late-stage roughly
    if x == -1:
        return 99
    return x-1



with open(f'{RESULTS_PATH}/fullroute/Early_imidazole_ring_formation.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=lambda x: abs(x-10)) # Early meanns early in the synthesis - deep in route
    

with open(f'{RESULTS_PATH}/fullroute/Late_imidazole_ring_formation.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=lambda x: abs(x-1))  # Late means late in the synthesis - shallow in route

with open(f'{RESULTS_PATH}/fullroute/No_ring_formation_reaction.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=lambda x: x == -1)  # No ring formation: bool

# 0...9 apply, 17...22. 25, 28, 29, 30, 32
with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._The_disconnection_should_be_made_between_two_pipe.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = SpecificBondBreak_1()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=route_score)


# With these ones we want the same, but prompt is different.
with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._One_intermediate_will_have_piperidine,_indole_and.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = SpecificBondBreak_1()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=route_score)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_intermediates._The_disconnection_should_be_made_between_diazepine_and_piperazine.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = SpecificBondBreak_2()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=route_score)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._The_disconnection_should_be_made_between_piperazi.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = SpecificBondBreak_3()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=route_score)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_where_the_key_disconnection_will_be_made_between_indole_and_amino-piperidine_rings.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = SpecificBondBreak_4()
gt_score, lmscore = ringbreaker_depth.depth_score(data, target_depth=route_score)


with open(f'{RESULTS_PATH}/fullroute/Form_piperidine-2,6-dione_and_oxoisoindolinone_rings_in_the_retrosynthesis._Get_the_piperidine_ring_.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond()
gt_score, length, lmscore = ringbreaker_depth.is_condition_met(data)

with open(f'{RESULTS_PATH}/fullroute/Form_piperidine_and_oxoisoindolinone_rings_in_the_synthesis._Get_the_piperidine-2,6-dione_from_comme.json', 'r') as f:
    data = json.load(f)
ringbreaker_depth = MultiRxnCond_single()
gt_score, length, lmscore = ringbreaker_depth.is_condition_met(data)

with open(f'{RESULTS_PATH}/fullroute/Form_only_oxoisoindolinone_ring_in_synthesis._Get_piperidine-2,6-dione_and_piperidine_rings_from_com.json', 'r') as f:
    data = json.load(f)
ringbreaker_depth = MultiRxnCond_single()
gt_score, length, lmscore = ringbreaker_depth.is_condition_met(data)

with open(f'{RESULTS_PATH}/fullroute/Form_piperidine_and_oxoisoindolinone_rings_in_the_synthesis._Make_sure_that_oxoisoindolinone_is_crea.json', 'r') as f:
    data = json.load(f)
ringbreaker_depth = MultiRxnCond_single()
gt_score, length, lmscore = ringbreaker_depth.is_condition_met(data)
with open(f'{RESULTS_PATH}/fullroute/Find_routes_with_no_ring_formations..json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond_single()
gt_score, length, lmscore = ringbreaker_depth.is_condition_met(data)