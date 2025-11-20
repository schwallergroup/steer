"""Generated evaluation code for: Azide intermediate for nitrogen introduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideIntermediatePresence(BaseScoring):
    """
    Checks for the presence of azide intermediates in synthesis routes.
    Azides serve as protected nitrogen precursors that can be reduced to amines.
    """
    
    def __init__(self, config: Dict):
        self.azide_pattern = config["parameters"]["smarts_pattern"]  # "[N-]=[N+]=[N-]"
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if x < 0:
                return 0  # Azide intermediate not found
            else:
                return 1  # Azide intermediate present
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))

    def hit_condition(self, d):
        """Check if this reaction node involves azide formation or utilization"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            products = parts[0]
            reactants = parts[1]
            
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(smi) for smi in products.split(".") if smi]
            react_mols = [Chem.MolFromSmiles(smi) for smi in reactants.split(".") if smi]
            
            if not all(prod_mols) or not all(react_mols):
                return False
            
            # Create azide pattern
            azide_mol = Chem.MolFromSmarts(self.azide_pattern)
            if azide_mol is None:
                return False
            
            # Check for azide formation (azide appears in products but not reactants)
            azide_in_products = any(mol.HasSubstructMatch(azide_mol) for mol in prod_mols)
            azide_in_reactants = any(mol.HasSubstructMatch(azide_mol) for mol in react_mols)
            
            # Return True if azide is formed or utilized in this step
            return azide_in_products or azide_in_reactants
            
        except Exception:
            return False
