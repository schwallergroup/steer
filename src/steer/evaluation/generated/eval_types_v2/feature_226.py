"""Generated evaluation code for: Late stage pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStagePyrazoleFormation(BaseScoring):
    """
    Evaluates whether pyrazole ring formation occurs late in the synthesis route.
    Rewards routes where pyrazole rings are formed in later stages rather than early.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ccnn1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.pyrazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage formation: later is better, so higher depth gets higher score.
        """
        if x < 0:
            return 0  # Pyrazole formation doesn't occur
        
        if self.timing == "late":
            return x * 10  # Later formation gets higher score
        else:
            return (1 - x) * 10  # Earlier formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves pyrazole ring formation.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
        
        try:
            # Split reaction into reactants and products
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
            
            products_smiles = parts[0]
            reactants_smiles = parts[1]
            
            # Parse molecules
            products = [Chem.MolFromSmiles(products_smiles)]
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            # Remove None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Count pyrazole rings in products and reactants
            product_pyrazoles = sum(len(mol.GetSubstructMatches(self.pyrazole_pattern)) 
                                  for mol in products)
            reactant_pyrazoles = sum(len(mol.GetSubstructMatches(self.pyrazole_pattern)) 
                                   for mol in reactants)
            
            # Check for pyrazole formation (more pyrazoles in products than reactants)
            if self.direction == "formation":
                return product_pyrazoles > reactant_pyrazoles
            else:  # ring breaking
                return reactant_pyrazoles > product_pyrazoles
                
        except Exception:
            return False
