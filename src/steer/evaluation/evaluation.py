"""Run evaluation script."""

# Benchmark stats
import json
from eval_types import *
# from steer.evaluation import _load_default_tasks

RESULTS_PATH = '../../../../synthegy/steer/'
PROMPT_TYPE = 'fullroute'


# task # defined in benchmark: smiles, prompt, eval_class
# Class needs to define eval method (e.g. what class to use for eval, and internally target_depth)

# tasks = _load_default_tasks()

# for task in tasks:
#     with open(f'{RESULTS_PATH}/{PROMPT_TYPE}/{task.id}.json', 'r') as f:
#         data = json.load(f)

#     evaluator = task.eval_class()
#     gt_score, lmscore = evaluator(data, task)



with open(f'{RESULTS_PATH}/fullroute/Early_imidazole_ring_formation.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth(config={"target_depth": {"type": "diff", "value": 10}})
gt_score, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)

with open(f'{RESULTS_PATH}/fullroute/Late_imidazole_ring_formation.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth(config={"target_depth": {"type": "diff", "value": 1}})
gt_score, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)

with open(f'{RESULTS_PATH}/fullroute/No_ring_formation_reaction.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = RingBreakDepth(config={"target_depth": {"type": "bool", "value": -1}})
gt_score, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._The_disconnection_should_be_made_between_two_pipe.json', 'r') as f:
    data = json.load(f)

ev = SpecificBondBreak(config={"bond_to_break": {"atom_1": 29, "atom_2": 33}})
gt_score, lmscore = ev(data)
print(gt_score, lmscore)


# With these ones we want the same, but prompt is different.
with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._One_intermediate_will_have_piperidine,_indole_and.json', 'r') as f:
    data = json.load(f)

ev = SpecificBondBreak(config={"bond_to_break": {"atom_1": 29, "atom_2": 33}})
gt_score, lmscore = ev(data)
print(gt_score, lmscore)

with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_intermediates._The_disconnection_should_be_made_between_diazepine_and_piperazine.json', 'r') as f:
    data = json.load(f)

ev = SpecificBondBreak(config={"bond_to_break": {"atom_1": 17, "atom_2": 21}})
gt_score, lmscore = ev(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_that_will_cut_the_molecule_in_two_similarly_sized_intermediates._The_disconnection_should_be_made_between_piperazi.json', 'r') as f:
    data = json.load(f)

ev = SpecificBondBreak(config={"bond_to_break": {"atom_1": 24, "atom_2": 26}})
gt_score, lmscore = ev(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute_no_feasibility/Identify_the_disconnection_strategy_where_the_key_disconnection_will_be_made_between_indole_and_amino-piperidine_rings.json', 'r') as f:
    data = json.load(f)

ev = SpecificBondBreak(config={"bond_to_break": {"atom_1": 36, "atom_2": 38}})
gt_score, lmscore = ev(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute/Form_piperidine-2,6-dione_and_oxoisoindolinone_rings_in_the_retrosynthesis._Get_the_piperidine_ring_.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond(config={"allow_piperidine": False, "allow_oxoisoindolinone": True, "allow_piperidine26diox": True})
gt_score, length, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)

with open(f'{RESULTS_PATH}/fullroute/Form_piperidine_and_oxoisoindolinone_rings_in_the_synthesis._Get_the_piperidine-2,6-dione_from_comme.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond(config={"allow_piperidine": True, "allow_oxoisoindolinone": True, "allow_piperidine26diox": False})
gt_score, length, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute/Form_only_oxoisoindolinone_ring_in_synthesis._Get_piperidine-2,6-dione_and_piperidine_rings_from_com.json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond(config={"allow_piperidine": False, "allow_oxoisoindolinone": True, "allow_piperidine26diox": False})
gt_score, length, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)

with open(f'{RESULTS_PATH}/fullroute/Form_piperidine_and_oxoisoindolinone_rings_in_the_synthesis._Make_sure_that_oxoisoindolinone_is_crea.json', 'r') as f:
    data = json.load(f)


ringbreaker_depth = MultiRxnCond(config={"allow_piperidine": True, "allow_oxoisoindolinone": True, "allow_piperidine26diox": False})
gt_score, length, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)


with open(f'{RESULTS_PATH}/fullroute/Find_routes_with_no_ring_formations..json', 'r') as f:
    data = json.load(f)

ringbreaker_depth = MultiRxnCond(config={"allow_piperidine": False, "allow_oxoisoindolinone": False, "allow_piperidine26diox": False})
gt_score, length, lmscore = ringbreaker_depth(data)
print(gt_score, lmscore)