"""Generated evaluation code for: Late stage lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LactamRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on late-stage lactam ring formation.
    Checks if a specific lactam ring is formed at a target depth in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_step = config["parameters"]["formation_step"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # For late-stage formation, prefer deeper in the tree (closer to 1.0)
            if x >= 0.5:  # Second half of synthesis
                return 10 * (1 - abs(x - 0.8))  # Optimal around 80% depth
            else:
                return 10 * x * 0.5  # Penalize early formation
        else:
            # For early-stage formation, prefer shallower depths
            return 10 * (1 - x)
    
    def hit_condition(self, d):
        """
        Check if this reaction step involves lactam ring formation.
        """
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
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if lactam ring is absent in reactants but present in products
            reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            products_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            # Ring formation occurs if ring is in products but not in reactants
            return products_have_ring and not reactants_have_ring
            
        except:
            return False
