"""Generated evaluation code for: Early piperidine ring saturation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyPiperidineRingSaturation(BaseScoring):
    """
    Evaluates whether piperidine ring formation/saturation occurs early in the synthesis.
    Checks for the presence of piperidine (C1CCNCC1) formation and rewards early occurrence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage = config["parameters"]["stage"]
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.piperidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Piperidine formation doesn't happen
        
        if self.stage == "early":
            # Reward early formation - lower depth fraction is better
            return 10 * (1 - x)  # x is depth fraction, so early (low x) gets high score
        else:
            # For late stage, higher depth fraction is better
            return 10 * x
    
    def hit_condition(self, d):
        """
        Checks if piperidine ring formation occurs in this reaction step.
        Looks for piperidine in products but not in all reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            # Parse products
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            products = [p for p in products if p is not None]
            
            # Parse reactants
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            # Check if any product contains piperidine
            product_has_piperidine = any(
                mol.HasSubstructMatch(self.piperidine_pattern) for mol in products
            )
            
            if not product_has_piperidine:
                return False
            
            # Check if piperidine formation is new (not present in all reactants)
            # This indicates ring formation/saturation occurred in this step
            reactant_has_piperidine = any(
                mol.HasSubstructMatch(self.piperidine_pattern) for mol in reactants
            )
            
            # Return True if piperidine appears in product but wasn't in reactants
            # OR if we're looking for saturation (tetrahydropyridine -> piperidine)
            return product_has_piperidine and not reactant_has_piperidine
            
        except Exception:
            return False
