"""Generated evaluation code for: Early spirocyclic cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySpirocyclicCyclopropaneFormation(BaseScoring):
    """
    Evaluates synthesis routes for early formation of spirocyclic cyclopropane rings.
    Detects cyclopropane ring formation reactions and rewards routes where this occurs
    early in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "early":
                return 1 - x  # Early formation is better (lower depth fraction)
            else:
                return x  # Late formation is better (higher depth fraction)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves cyclopropane ring formation.
        Returns True if cyclopropane rings are formed in this step.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse products and reactants
            prod_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".") if smi]
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".") if smi]
            
            if not all(prod_mols) or not all(react_mols):
                return False
            
            # Count cyclopropane rings in products and reactants
            prod_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                for mol in prod_mols if mol is not None)
            react_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                 for mol in react_mols if mol is not None)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return prod_ring_count > react_ring_count
            elif self.direction == "breaking":
                return react_ring_count > prod_ring_count
            else:
                return prod_ring_count != react_ring_count
                
        except Exception:
            return False
