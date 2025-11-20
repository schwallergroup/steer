"""Generated evaluation code for: Early Fmoc protection of secondary amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyFmocProtection(BaseScoring):
    """
    Evaluates if Fmoc protection of secondary amine occurs early in the synthesis route.
    
    Detects reactions where Fmoc group is introduced to protect secondary amines,
    and scores based on how early this protection occurs in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        
    def route_scoring(self, x) -> float:
        """
        Scores based on timing of Fmoc protection.
        Early protection (lower depth fraction) gets higher score.
        """
        if x < 0:
            return 0  # No Fmoc protection found
        
        if self.timing_preference == "early":
            return 1 - x  # Reward early protection (lower depth fraction)
        else:
            return x  # Reward late protection if specified
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the reaction introduces Fmoc protection on a secondary amine.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check if Fmoc reagent is present in reactants
            fmoc_patterns = [
                "O=C(Oc1ccc2c(c1)C(C)(C)OC21CCc2ccccc21)N",  # Fmoc-Cl
                "O=C(ON1C(=O)CCC1=O)N2C(=O)c3ccccc3C3=C2C=CC=C3",  # Fmoc-OSu
                "FC(F)(F)C(=O)Oc1ccc2c(c1)C(C)(C)OC21CCc2ccccc21"  # Alternative Fmoc reagent
            ]
            
            has_fmoc_reagent = False
            for reactant in reactants:
                for pattern in fmoc_patterns:
                    fmoc_mol = Chem.MolFromSmarts(pattern)
                    if fmoc_mol and reactant.HasSubstructMatch(fmoc_mol):
                        has_fmoc_reagent = True
                        break
                if has_fmoc_reagent:
                    break
            
            # Check for secondary amine in reactants and Fmoc-protected amine in products
            secondary_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([CX4])([CX4])")
            fmoc_protected_pattern = Chem.MolFromSmarts("O=C(N)Oc1ccc2c(c1)C(C)(C)OC21CCc2ccccc21")
            
            # Check if reactants contain secondary amine
            has_secondary_amine = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(secondary_amine_pattern):
                    has_secondary_amine = True
                    break
            
            # Check if products contain Fmoc-protected amine
            has_fmoc_product = False
            for product in products:
                if product.HasSubstructMatch(fmoc_protected_pattern):
                    has_fmoc_product = True
                    break
            
            return has_fmoc_reagent and has_secondary_amine and has_fmoc_product
            
        except Exception:
            return False
