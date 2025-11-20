"""Generated evaluation code for: Multiple protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes based on protecting group cycling strategy.
    Counts protection/deprotection cycles for specified functional groups.
    """
    
    def __init__(self, config):
        self.min_cycles = int(config["protection_deprotection_cycles"].replace(">", ""))
        self.functional_group = config["functional_group"]
        
        # Define protecting group patterns for different functional groups
        self.protecting_patterns = {
            "amine": {
                "boc": "[NH1,NH2][C](=O)OC(C)(C)C",
                "cbz": "[NH1,NH2][C](=O)Oc1ccccc1",
                "benzyl": "[NH1,NH2]Cc1ccccc1",
                "fmoc": "[NH1,NH2][C](=O)OCC1c2ccccc2-c2ccccc21",
                "tosyl": "[NH1,NH2][S](=O)(=O)c1ccc(C)cc1"
            },
            "alcohol": {
                "tbdms": "[OH1]Si(C)(C)C(C)(C)C",
                "acetyl": "[OH1][C](=O)C",
                "benzyl": "[OH1]Cc1ccccc1"
            },
            "carboxylic_acid": {
                "methyl_ester": "[C](=O)OC",
                "ethyl_ester": "[C](=O)OCC",
                "tert_butyl_ester": "[C](=O)OC(C)(C)C"
            }
        }
    
    def condition_depth(self, d):
        reactions = self.get_rxns(d)
        cycles = self.count_protection_cycles(reactions)
        condition = cycles > self.min_cycles
        return condition, len(reactions)
    
    def count_protection_cycles(self, reactions):
        """Count complete protection/deprotection cycles"""
        if self.functional_group not in self.protecting_patterns:
            return 0
            
        patterns = self.protecting_patterns[self.functional_group]
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            rxn_smiles = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                continue
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                continue
                
            reactants_smiles = parts[0]
            products_smiles = parts[1]
            
            try:
                reactants = [Chem.MolFromSmiles(s.strip()) for s in reactants_smiles.split(".")]
                products = [Chem.MolFromSmiles(s.strip()) for s in products_smiles.split(".")]
                
                reactants = [mol for mol in reactants if mol is not None]
                products = [mol for mol in products if mol is not None]
                
                # Check for protection (protecting group appears in products)
                for pg_name, pattern_smarts in patterns.items():
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern is None:
                        continue
                        
                    reactant_matches = sum(1 for mol in reactants if mol.HasSubstructMatch(pattern))
                    product_matches = sum(1 for mol in products if mol.HasSubstructMatch(pattern))
                    
                    if product_matches > reactant_matches:
                        # Protection event
                        protection_events.append({
                            'step': i,
                            'type': 'protection',
                            'group': pg_name,
                            'count': product_matches - reactant_matches
                        })
                    elif reactant_matches > product_matches:
                        # Deprotection event
                        protection_events.append({
                            'step': i,
                            'type': 'deprotection', 
                            'group': pg_name,
                            'count': reactant_matches - product_matches
                        })
                        
            except Exception:
                continue
        
        # Count complete cycles
        cycles = 0
        group_stacks = {}
        
        for event in protection_events:
            group = event['group']
            if group not in group_stacks:
                group_stacks[group] = 0
                
            if event['type'] == 'protection':
                group_stacks[group] += event['count']
            elif event['type'] == 'deprotection':
                # Each deprotection can complete a cycle if there was a prior protection
                deprotections = min(event['count'], group_stacks[group])
                cycles += deprotections
                group_stacks[group] -= deprotections
                
        return cycles
    
    def route_scoring(self, x):
        """Score based on number of protection cycles"""
        if x < 0:
            return 0  # No cycles found
        # Higher number of cycles gets better score (up to reasonable limit)
        return min(10, x * 2)
