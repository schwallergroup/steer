"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation timing.
    Detects when a specified ring structure is formed and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score
        else:  # early
            return x  # Earlier formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves the target ring formation/break"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Remove None molecules from parsing failures
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count ring matches in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                 for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            else:  # break
                # Ring breaking: fewer rings in products than reactants
                return reactant_matches > product_matches
                
        except Exception:
            return False
