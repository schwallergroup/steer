"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring structure is formed late in the synthesis route.
    Detects ring formation by checking if the target ring is present in products but 
    not in reactants of a reaction.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config.get("timing", "late")  # "early", "late", or specific depth
        self.direction = config.get("direction", "formation")  # "formation" or "break"
        
        # Compile the SMARTS pattern for efficiency
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if self.ring_pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.ring_smarts}")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later is better, x is depth fraction (0=root, 1=leaves)
        elif self.timing == "early":
            return x  # Earlier is better
        else:
            # Specific timing target (assume it's a depth fraction)
            target_depth = float(self.timing)
            return max(0, 1 - abs(x - target_depth))

    def hit_condition(self, d) -> bool:
        """
        Check if the target ring is formed (or broken) in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse products
            products = []
            for prod_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(prod_smiles)
                if mol is not None:
                    products.append(mol)
            
            # Parse reactants
            reactants = []
            for react_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(react_smiles)
                if mol is not None:
                    reactants.append(mol)
            
            # Check ring presence in products and reactants
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            if self.direction == "formation":
                # Ring formation: present in products but not in reactants
                return ring_in_products and not ring_in_reactants
            elif self.direction == "break":
                # Ring breaking: present in reactants but not in products
                return ring_in_reactants and not ring_in_products
            
        except Exception:
            return False
        
        return False
