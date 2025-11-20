"""Generated evaluation code for: Late pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation timing.
    Detects when a specific ring (pyridine by default) is formed and 
    scores based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "breaking"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation/breaking doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, so higher depth fraction gets lower score
        else:  # early
            return x  # Earlier is better, so lower depth fraction gets lower score
    
    def hit_condition(self, d):
        """Check if this reaction involves formation/breaking of the target ring"""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Parse molecules
        prod_mols = [Chem.MolFromSmiles(products)]
        react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
        
        if None in prod_mols or None in react_mols:
            return False
            
        # Create pattern matcher
        ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if ring_pattern is None:
            return False
        
        # Count ring occurrences in products and reactants
        prod_ring_count = sum(len(mol.GetSubstructMatches(ring_pattern)) 
                             for mol in prod_mols if mol is not None)
        react_ring_count = sum(len(mol.GetSubstructMatches(ring_pattern)) 
                              for mol in react_mols if mol is not None)
        
        if self.direction == "formation":
            # Ring formation: more rings in products than reactants
            return prod_ring_count > react_ring_count
        else:  # breaking
            # Ring breaking: fewer rings in products than reactants
            return prod_ring_count < react_ring_count
