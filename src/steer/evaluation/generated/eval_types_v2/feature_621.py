"""Generated evaluation code for: Late oxazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OxazoleRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on when oxazole ring formation occurs.
    Rewards late-stage oxazole formation (closer to final product).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1ncco1"
        self.timing = config["parameters"]["timing"]  # "late" 
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (lower depth fraction)
            elif self.timing == "early":
                return x  # Earlier formation is better (higher depth fraction)
            else:
                return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d) -> bool:
        """
        Check if oxazole ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count oxazole rings in reactants and products
            reactant_oxazole_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                       for mol in reactants)
            product_oxazole_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                      for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more oxazole rings in products than reactants
                return product_oxazole_count > reactant_oxazole_count
            elif self.direction == "breaking":
                # Ring breaking: fewer oxazole rings in products than reactants
                return reactant_oxazole_count > product_oxazole_count
            else:
                # Any change in oxazole ring count
                return reactant_oxazole_count != product_oxazole_count
                
        except Exception:
            return False
