"""Generated evaluation code for: Late stage triazolone ring formation via convergent coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoloneRingFormation(BaseScoring):
    """
    Evaluates routes based on late-stage triazolone ring formation via convergent coupling.
    Checks if a triazolone ring ([nH]1c(=O)n[nH]c1) is formed at a specified timing threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.step_threshold = config["parameters"]["step_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # For late-stage formation, higher depth fraction is better
            # Convert to 0-10 scale where later formation scores higher
            if x >= self.step_threshold:
                return 8 + (x - self.step_threshold) * 2 / (1 - self.step_threshold)  # 8-10 range
            else:
                return x * 8 / self.step_threshold  # 0-8 range
        elif self.timing == "early":
            # For early-stage formation, lower depth fraction is better
            return 10 * (1 - x)
        else:
            # Default case - any formation is good
            return 8.0
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a triazolone ring by convergent coupling.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
            
            # Check if product contains triazolone ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check reactants to see if triazolone ring is absent
            reactant_smiles_list = reactants_smiles.split(".")
            if len(reactant_smiles_list) < 2:
                return False  # Not convergent if less than 2 reactants
            
            # Verify that no reactant already contains the triazolone ring
            for reactant_smiles in reactant_smiles_list:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already present in reactant
            
            # If we reach here: product has triazolone, reactants don't, and it's convergent
            return True
            
        except Exception:
            return False
