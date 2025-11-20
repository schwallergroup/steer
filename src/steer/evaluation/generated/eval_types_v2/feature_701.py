"""Generated evaluation code for: Early Suzuki coupling for biaryl scaffold"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs early in the synthesis route.
    Checks for the presence of Suzuki coupling (biaryl formation) at or before a specified depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.timing = config["parameters"]["timing"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        else:
            # Early coupling is better - score decreases with depth
            if x <= self.depth_threshold / 10.0:  # x is depth fraction
                return 10 - (x * 10)  # Full score for very early coupling
            else:
                return max(0, 10 - (x * 15))  # Penalty for late coupling
    
    def hit_condition(self, d):
        """
        Detect Suzuki coupling by looking for:
        1. Biaryl bond formation (two aromatic rings connected)
        2. Presence of boron-containing reagent in reactants
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for boron-containing reactant (organoborane)
            has_boron_reagent = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts("[B]")) for mol in reactants
            )
            
            # Check for biaryl formation - look for newly formed aromatic C-C bonds
            if has_boron_reagent:
                # Pattern for biaryl: aromatic carbon connected to aromatic carbon
                biaryl_pattern = Chem.MolFromSmarts("c-c")
                
                if product.HasSubstructMatch(biaryl_pattern):
                    # Verify this is a new bond by checking reactants don't have this specific connection
                    return self._verify_new_biaryl_bond(reactants, product)
            
            return False
            
        except Exception:
            return False
    
    def _verify_new_biaryl_bond(self, reactants, product):
        """
        Verify that a biaryl bond was actually formed in this reaction
        by checking that the biaryl substructure is not present in individual reactants
        """
        # Look for extended biaryl patterns that would indicate coupling
        extended_biaryl_patterns = [
            "c1ccccc1-c2ccccc2",  # Simple biphenyl
            "c1ccc(cc1)-c2ccccc2",  # Para-substituted biphenyl
            "c1cc(ccc1)-c2ccccc2",  # Meta-substituted biphenyl
        ]
        
        for pattern_smarts in extended_biaryl_patterns:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and product.HasSubstructMatch(pattern):
                # Check that this pattern is not in any single reactant
                pattern_in_reactants = any(
                    reactant.HasSubstructMatch(pattern) for reactant in reactants
                )
                if not pattern_in_reactants:
                    return True
        
        return False
