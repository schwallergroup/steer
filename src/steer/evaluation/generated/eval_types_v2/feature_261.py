"""Generated evaluation code for: Late stage lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed at a late stage in the synthesis.
    Checks for the formation of a lactam ring pattern at depths beyond the stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # For late-stage formation, reward formations that occur after stage_threshold
            if x >= self.stage_threshold:
                return 10 * (1 - (1 - x) / (1 - self.stage_threshold))  # Scale 0-10
            else:
                return 0  # Too early
        elif self.timing == "early":
            # For early-stage formation, reward formations before stage_threshold
            if x <= self.stage_threshold:
                return 10 * (1 - x / self.stage_threshold)  # Scale 0-10
            else:
                return 0  # Too late
        else:
            # Default: any formation is good, but later is better for "late" timing
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """Check if the target ring is formed in this reaction step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check if ring pattern is absent in reactants but present in products
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            # Ring formation occurs if pattern is not in reactants but is in products
            return not ring_in_reactants and ring_in_products
            
        except Exception:
            return False
