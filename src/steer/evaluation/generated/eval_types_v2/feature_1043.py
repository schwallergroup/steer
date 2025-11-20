"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates late-stage formation of a specific ring structure.
    Looks for reactions where the target ring is formed and rewards
    later occurrence in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, score decreases with earlier occurrence
        else:  # early
            return x  # Earlier is better, score increases with later occurrence
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves formation of the target ring"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create pattern from SMARTS
            pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pattern is None:
                return False
            
            # Count ring occurrences in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in reactants)
            product_matches = sum(len(mol.GetSubstructMatches(pattern)) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_matches > reactant_matches
            else:  # "break"
                # Ring breaking: fewer rings in products than reactants  
                return reactant_matches > product_matches
                
        except Exception:
            return False
