"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStage_PyrazoleRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrazole ring formation.
    Detects when pyrazole rings are formed in the latter stages of synthesis,
    typically via cyclization reactions like Knorr synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config.get("ring_smarts", "c1ccnnc1")  # pyrazole pattern
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late-stage preference, higher depth fraction is better
            # Convert to 0-10 scale where late-stage gets higher scores
            if self.timing == "late":
                return x * 10  # Later formation scores higher
            else:
                return (1 - x) * 10  # Earlier formation scores higher
    
    def hit_condition(self, d) -> bool:
        """
        Checks if pyrazole ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create pyrazole pattern matcher
            pyrazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if pyrazole_pattern is None:
                return False
            
            # Count pyrazole rings in reactants and products
            reactant_pyrazoles = sum(len(mol.GetSubstructMatches(pyrazole_pattern)) 
                                   for mol in reactants)
            product_pyrazoles = sum(len(mol.GetSubstructMatches(pyrazole_pattern)) 
                                  for mol in products)
            
            # Check for ring formation (more pyrazoles in products than reactants)
            if self.direction == "formation":
                return product_pyrazoles > reactant_pyrazoles
            elif self.direction == "breaking":
                return reactant_pyrazoles > product_pyrazoles
            
            return False
            
        except Exception:
            return False
