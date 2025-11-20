"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling reactions occur in the late stages of synthesis.
    Detects amide bond formation using SMARTS pattern and rewards late-stage occurrence.
    """
    
    def __init__(self, config: Dict):
        self.reaction_smarts = config["parameters"]["reaction_smarts"]
        self.timing = config["parameters"]["timing"]
        self.bond_formation = config["parameters"]["bond_formation"]
        self.amide_pattern = Chem.MolFromSmarts(self.reaction_smarts)

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't occur
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage coupling is better (lower depth fraction = higher score)
            else:
                return x  # Early-stage coupling preferred

    def hit_condition(self, d) -> bool:
        """Check if this reaction involves amide bond formation"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants_smiles = parts[0]
            products_smiles = parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check if amide pattern is present in products but not in reactants
            # (indicating amide bond formation)
            if self.bond_formation:
                # Check if amide is formed (present in products, not in reactants)
                amide_in_products = any(mol.HasSubstructMatch(self.amide_pattern) for mol in products)
                amide_in_reactants = any(mol.HasSubstructMatch(self.amide_pattern) for mol in reactants)
                
                # True if amide bond is formed in this step
                return amide_in_products and not amide_in_reactants
            else:
                # Just check presence in products
                return any(mol.HasSubstructMatch(self.amide_pattern) for mol in products)
                
        except Exception:
            return False
