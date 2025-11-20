"""Generated evaluation code for: Late triazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoleRingFormation(BaseScoring):
    """
    Evaluates the timing of triazole ring formation in synthesis routes.
    Rewards late-stage triazole formation where the triazole ring is formed
    in the final steps from precursors on a complex scaffold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" 
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation gets higher score
            else:
                return x  # Earlier formation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a triazole ring formation occurs in this reaction step.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Create triazole pattern
            triazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if triazole_pattern is None:
                return False
            
            # Count triazole rings in reactants and products
            reactant_triazoles = sum(len(mol.GetSubstructMatches(triazole_pattern)) 
                                   for mol in reactants)
            product_triazoles = sum(len(mol.GetSubstructMatches(triazole_pattern)) 
                                  for mol in products)
            
            # Check for ring formation (more triazoles in products than reactants)
            if self.direction == "formation":
                return product_triazoles > reactant_triazoles
            else:  # ring breaking
                return reactant_triazoles > product_triazoles
                
        except (KeyError, IndexError, AttributeError):
            return False
