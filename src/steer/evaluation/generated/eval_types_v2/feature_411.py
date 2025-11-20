"""Generated evaluation code for: Tetrazole ring formation from nitrile"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TetrazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for tetrazole ring formation from nitrile via azide-nitrile cycloaddition.
    
    This scorer identifies [3+2] cycloaddition reactions between azides and nitriles that form
    tetrazole rings, rewarding earlier occurrence in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Tetrazole formation doesn't happen
        else:
            return 1 - x  # Earlier tetrazole formation is better
    
    def hit_condition(self, d):
        """Check if this reaction node represents tetrazole formation from nitrile and azide."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains tetrazole ring
            tetrazole_pattern = Chem.MolFromSmarts("[#6]-1-[#7]=[#7]-[#7]=[#7]-1")  # tetrazole ring
            if not product.HasSubstructMatch(tetrazole_pattern):
                return False
            
            # Check if reactants contain nitrile and azide
            nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")  # C≡N
            azide_pattern = Chem.MolFromSmarts("[#7]=[#7]#[#7]")  # N=N≡N
            
            has_nitrile = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
            has_azide = any(mol.HasSubstructMatch(azide_pattern) for mol in reactants)
            
            # Verify this is a cycloaddition (reactants don't have tetrazole, product does)
            reactants_have_tetrazole = any(mol.HasSubstructMatch(tetrazole_pattern) for mol in reactants)
            
            return has_nitrile and has_azide and not reactants_have_tetrazole
            
        except Exception:
            return False
