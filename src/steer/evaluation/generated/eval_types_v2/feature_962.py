"""Generated evaluation code for: Late stage pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation reactions.
    Checks if a specific ring (defined by SMARTS pattern) is formed late in the synthesis.
    Returns higher scores for ring formation that occurs deeper in the route tree.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config.get("timing", "late")  # "early", "late", or specific depth
        self.direction = config.get("direction", "formation")  # "formation" or "breaking"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        For late-stage formation, higher depth fractions get higher scores.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Late-stage formation preferred: score increases with depth
            return 10 * x
        elif self.timing == "early":
            # Early-stage formation preferred: score decreases with depth
            return 10 * (1 - x)
        else:
            # Specific timing target (assuming numeric value)
            target_depth = float(self.timing)
            return max(0, 10 - 10 * abs(x - target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves the target ring formation/breaking.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Count ring occurrences in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                # Ring breaking: fewer rings in products than reactants
                return reactant_rings > product_rings
            else:
                return False
                
        except Exception:
            return False
