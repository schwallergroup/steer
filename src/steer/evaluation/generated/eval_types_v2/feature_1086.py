"""Generated evaluation code for: Late oxadiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoleFormation(BaseScoring):
    """
    Evaluates whether oxadiazole ring formation occurs late in the synthesis.
    Detects when an oxadiazole ring (c1onc[nH]1) is formed and rewards later formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.oxadiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is rewarded (higher x is better)
            return x * 10  # Convert fraction to 0-10 score
    
    def hit_condition(self, d) -> bool:
        """
        Check if oxadiazole ring formation occurs in this reaction step.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        # Parse reactants and products
        reactants_smiles = rxn_parts[0].split(".")
        products_smiles = rxn_parts[1].split(".")
        
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles if smi]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles if smi]
            
            if not all(reactants) or not all(products):
                return False
            
            # Count oxadiazole rings in reactants and products
            reactant_oxadiazole_count = sum(
                len(mol.GetSubstructMatches(self.oxadiazole_pattern)) 
                for mol in reactants
            )
            
            product_oxadiazole_count = sum(
                len(mol.GetSubstructMatches(self.oxadiazole_pattern)) 
                for mol in products
            )
            
            # Ring formation: more oxadiazole rings in products than reactants
            if self.direction == "formation":
                return product_oxadiazole_count > reactant_oxadiazole_count
            else:  # breaking
                return reactant_oxadiazole_count > product_oxadiazole_count
                
        except Exception:
            return False
