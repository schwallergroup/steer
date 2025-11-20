"""Generated evaluation code for: Late stage piperidine ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingClosure(BaseScoring):
    """
    Evaluates whether a specific ring closure occurs at late stage in the synthesis.
    Checks for the formation of a specified ring pattern and rewards later occurrence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring closure doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, score decreases with earlier depth
        elif self.timing == "early":
            return x  # Earlier is better, score increases with later depth
        else:
            return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d) -> bool:
        """Check if the specified ring closure occurs in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            if self.direction == "formation":
                # Check if ring is formed: present in product but not in any single reactant
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                if not product_has_ring:
                    return False
                
                # Ring should not be present as complete structure in any single reactant
                for reactant in reactant_mols:
                    if reactant.HasSubstructMatch(self.ring_pattern):
                        return False
                
                return True
                
            elif self.direction == "breaking":
                # Check if ring is broken: present in reactant but not in products
                reactant_has_ring = any(r.HasSubstructMatch(self.ring_pattern) for r in reactant_mols)
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                return reactant_has_ring and not product_has_ring
                
        except Exception:
            return False
        
        return False
