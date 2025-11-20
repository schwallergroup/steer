"""Generated evaluation code for: Late piperazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePiperazineFormation(BaseScoring):
    """
    Evaluates whether piperazine ring formation occurs late in the synthesis route.
    Looks for cyclization reactions that form the piperazine ring structure.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage formation is better (higher score for larger x)
            else:  # early timing
                return x  # Early-stage formation is better (higher score for smaller x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves piperazine ring formation"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Create piperazine pattern
            piperazine_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if piperazine_pattern is None:
                return False
            
            # Check if this is ring formation (reactants don't have ring, products do)
            if self.direction == "formation":
                reactants_have_ring = any(mol.HasSubstructMatch(piperazine_pattern) for mol in reactants)
                products_have_ring = any(mol.HasSubstructMatch(piperazine_pattern) for mol in products)
                
                return not reactants_have_ring and products_have_ring
            
            # Check if this is ring breaking (reactants have ring, products don't)
            elif self.direction == "breaking":
                reactants_have_ring = any(mol.HasSubstructMatch(piperazine_pattern) for mol in reactants)
                products_have_ring = any(mol.HasSubstructMatch(piperazine_pattern) for mol in products)
                
                return reactants_have_ring and not products_have_ring
                
        except Exception:
            return False
            
        return False
