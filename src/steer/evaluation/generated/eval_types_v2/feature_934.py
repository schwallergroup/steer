"""Generated evaluation code for: Late purine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePurineRingFormation(BaseScoring):
    """
    Evaluates routes based on when purine ring formation occurs.
    Rewards routes where purine ring formation happens late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.purine_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Purine ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (lower depth fraction)
            elif self.timing == "early":
                return x  # Earlier formation is better (higher depth fraction)
            else:
                return 1 if x >= 0 else 0  # Just check if it happens
    
    def hit_condition(self, d):
        """Check if purine ring formation occurs in this reaction step."""
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        # Parse molecules
        try:
            prod_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".")]
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".")]
            
            if not all(mol for mol in prod_mols + react_mols):
                return False
                
            purine_pattern = Chem.MolFromSmarts(self.purine_smarts)
            if not purine_pattern:
                return False
                
            # Check for purine ring formation
            if self.direction == "formation":
                # Purine present in products but not in reactants
                prod_has_purine = any(mol.HasSubstructMatch(purine_pattern) for mol in prod_mols)
                react_has_purine = any(mol.HasSubstructMatch(purine_pattern) for mol in react_mols)
                
                return prod_has_purine and not react_has_purine
            
            elif self.direction == "break":
                # Purine present in reactants but not in products
                prod_has_purine = any(mol.HasSubstructMatch(purine_pattern) for mol in prod_mols)
                react_has_purine = any(mol.HasSubstructMatch(purine_pattern) for mol in react_mols)
                
                return react_has_purine and not prod_has_purine
                
        except Exception:
            return False
            
        return False
