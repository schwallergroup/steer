"""Generated evaluation code for: Multiple ester protecting group cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects multiple protection-deprotection cycles of carboxylic acids using different ester protecting groups.
    Checks if the same carboxylic acid undergoes protection with ethyl and methyl esters in cycles.
    """
    
    def __init__(self, config):
        self.protecting_groups = config["parameters"]["protecting_groups"]
        self.required_cycle_count = config["parameters"]["cycle_count"]
        
        # SMARTS patterns for ester formation and hydrolysis
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        self.ethyl_ester_pattern = Chem.MolFromSmarts("[C](=O)[O][CH2][CH3]")
        self.methyl_ester_pattern = Chem.MolFromSmarts("[C](=O)[O][CH3]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection/deprotection events by carbon atom map
        protection_events = {}  # {carbon_map_num: [event_type, ...]}
        
        for rxn in reactions:
            events = self.detect_ester_protection_events(rxn)
            for carbon_map, event_type in events:
                if carbon_map not in protection_events:
                    protection_events[carbon_map] = []
                protection_events[carbon_map].append(event_type)
        
        # Check for cycling pattern
        valid_cycles = 0
        for carbon_map, events in protection_events.items():
            cycles = self.count_protection_cycles(events)
            if cycles >= self.required_cycle_count:
                valid_cycles += 1
        
        condition_met = valid_cycles > 0
        return condition_met, len(reactions)
    
    def detect_ester_protection_events(self, rxn):
        """
        Detects ester protection/deprotection events in a reaction.
        Returns list of (carbon_atom_map, event_type) tuples.
        """
        events = []
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return events
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Get all molecules
            all_reactants = [mol for mol in reactant_mols if mol is not None]
            all_products = [mol for mol in product_mols if mol is not None]
            
            # Track carbonyl carbons and their states
            reactant_carbonyls = self.get_carbonyl_states(all_reactants)
            product_carbonyls = self.get_carbonyl_states(all_products)
            
            # Detect state changes
            for carbon_map in set(reactant_carbonyls.keys()) | set(product_carbonyls.keys()):
                reactant_state = reactant_carbonyls.get(carbon_map, "none")
                product_state = product_carbonyls.get(carbon_map, "none")
                
                if reactant_state != product_state:
                    if reactant_state == "carboxylic_acid" and product_state in ["methyl_ester", "ethyl_ester"]:
                        events.append((carbon_map, f"protect_{product_state}"))
                    elif product_state == "carboxylic_acid" and reactant_state in ["methyl_ester", "ethyl_ester"]:
                        events.append((carbon_map, f"deprotect_{reactant_state}"))
                        
        except Exception:
            pass
            
        return events
    
    def get_carbonyl_states(self, molecules):
        """
        Returns a dictionary mapping carbonyl carbon atom map numbers to their protection states.
        """
        carbonyl_states = {}
        
        for mol in molecules:
            if mol is None:
                continue
                
            # Find carboxylic acids
            acid_matches = mol.GetSubstructMatches(self.carboxylic_acid_pattern)
            for match in acid_matches:
                carbon_atom = mol.GetAtomWithIdx(match[0])
                if carbon_atom.GetAtomMapNum() > 0:
                    carbonyl_states[carbon_atom.GetAtomMapNum()] = "carboxylic_acid"
            
            # Find methyl esters
            methyl_matches = mol.GetSubstructMatches(self.methyl_ester_pattern)
            for match in methyl_matches:
                carbon_atom = mol.GetAtomWithIdx(match[0])
                if carbon_atom.GetAtomMapNum() > 0:
                    carbonyl_states[carbon_atom.GetAtomMapNum()] = "methyl_ester"
            
            # Find ethyl esters
            ethyl_matches = mol.GetSubstructMatches(self.ethyl_ester_pattern)
            for match in ethyl_matches:
                carbon_atom = mol.GetAtomWithIdx(match[0])
                if carbon_atom.GetAtomMapNum() > 0:
                    carbonyl_states[carbon_atom.GetAtomMapNum()] = "ethyl_ester"
                    
        return carbonyl_states
    
    def count_protection_cycles(self, events):
        """
        Counts the number of complete protection-deprotection cycles using different protecting groups.
        """
        if len(events) < 4:  # Need at least protect->deprotect->protect->deprotect
            return 0
            
        cycles = 0
        i = 0
        used_groups = set()
        
        while i < len(events) - 3:
            # Look for pattern: protect_X -> deprotect_X -> protect_Y -> deprotect_Y
            if (events[i].startswith("protect_") and 
                events[i+1] == events[i].replace("protect_", "deprotect_") and
                events[i+2].startswith("protect_") and
                events[i+3] == events[i+2].replace("protect_", "deprotect_")):
                
                group1 = events[i].split("_")[1]
                group2 = events[i+2].split("_")[1]
                
                # Must use different protecting groups
                if group1 != group2 and group1 in ["methyl_ester", "ethyl_ester"] and group2 in ["methyl_ester", "ethyl_ester"]:
                    cycles += 1
                    used_groups.add(group1)
                    used_groups.add(group2)
                    i += 4
                else:
                    i += 1
            else:
                i += 1
                
        return cycles
