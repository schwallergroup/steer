"""Generated evaluation code for: Late imidazopyrazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ImidazopyrazineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage imidazopyrazine ring formation.
    
    This class checks if the imidazopyrazine bicyclic system (n1ccnc2ncccc12) 
    is formed in the final cyclization step of the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.step_position = config["parameters"]["step_position"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        elif self.timing == "late" and self.step_position == "final":
            # For final step requirement, only depth 0 (final step) gets full score
            if x == 0:
                return 1.0
            else:
                return 0  # Ring formation not in final step
        elif self.timing == "late":
            # Late-stage formation is better (lower depth fraction preferred)
            return 1 - x
        else:
            return 1 - x
    
    def hit_condition(self, d):
        """Check if imidazopyrazine ring is formed in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactant_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            # Check if product contains the imidazopyrazine ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant lacks the complete imidazopyrazine system
            reactants = reactant_smiles.split(".")
            for reactant_smi in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smi)
                if reactant_mol and not reactant_mol.HasSubstructMatch(self.ring_pattern):
                    # At least one reactant lacks the ring system, so ring is being formed
                    return True
            
            return False  # All reactants already have the ring system
            
        except Exception:
            return False
