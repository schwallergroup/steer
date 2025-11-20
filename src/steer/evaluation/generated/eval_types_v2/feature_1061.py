"""Generated evaluation code for: Multiple ester protection cycling steps"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterProtectionCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multiple ester protection cycling steps.
    Detects repeated formation/hydrolysis of ester protecting groups across the route.
    """
    
    def __init__(self, config):
        self.protecting_group_smarts = config.get("protecting_group_smarts", "[C](=O)[O][C]")
        self.target_cycle_count = config.get("cycle_count", 3)
        self.strategy = config.get("strategy", "cycling")
        self.ester_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Analyzes the entire route tree to count ester protection cycling events.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        cycle_events = self.count_ester_cycles(reactions)
        
        condition = cycle_events >= self.target_cycle_count
        return condition, len(reactions)
    
    def count_ester_cycles(self, reactions) -> int:
        """
        Count the number of ester protection/deprotection cycling events.
        A cycle is defined as formation followed by hydrolysis (or vice versa) of esters.
        """
        ester_events = []
        
        # Analyze each reaction for ester formation/hydrolysis
        for rxn in reactions:
            event_type = self.classify_ester_reaction(rxn)
            if event_type:
                ester_events.append(event_type)
        
        # Count cycling patterns (alternating formation/hydrolysis)
        cycle_count = 0
        for i in range(len(ester_events) - 1):
            if ester_events[i] != ester_events[i + 1]:
                cycle_count += 1
        
        return cycle_count
    
    def classify_ester_reaction(self, rxn) -> str:
        """
        Classify reaction as ester formation, hydrolysis, or neither.
        Returns 'formation', 'hydrolysis', or None.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Count ester groups in reactants and products
            reactant_esters = self.count_ester_groups(reactants_smiles)
            product_esters = self.count_ester_groups(products_smiles)
            
            if product_esters > reactant_esters:
                return "formation"
            elif reactant_esters > product_esters:
                return "hydrolysis"
            else:
                return None
                
        except Exception:
            return None
    
    def count_ester_groups(self, smiles) -> int:
        """Count total number of ester groups in all molecules in SMILES string."""
        total_esters = 0
        
        # Handle multiple molecules separated by dots
        for mol_smiles in smiles.split("."):
            mol = Chem.MolFromSmiles(mol_smiles.strip())
            if mol:
                matches = mol.GetSubstructMatches(self.ester_pattern)
                total_esters += len(matches)
        
        return total_esters
    
    def route_scoring(self, x) -> float:
        """
        Convert cycle count to 0-10 score.
        Higher scores for routes meeting the target cycle count.
        """
        if x >= self.target_cycle_count:
            return 10.0  # Perfect score for meeting cycle requirement
        elif x > 0:
            return 5.0 + (x / self.target_cycle_count) * 5.0  # Partial credit
        else:
            return 0.0  # No cycling detected
