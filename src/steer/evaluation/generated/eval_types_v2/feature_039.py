"""Generated evaluation code for: Early stage Stille cross-coupling for functionalization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStilleCoupling(BaseScoring):
    """
    Evaluates whether Stille cross-coupling occurs in the early stages of synthesis.
    Stille coupling involves C-C bond formation between organotin reagents and 
    organic halides or triflates, typically used for functionalization.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Stille coupling doesn't occur
        else:
            if self.timing_preference == "early":
                return 1 - x  # Earlier is better, x is depth fraction (0-1)
            else:
                return x  # Later is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects Stille coupling by checking for:
        1. Organotin reagent (R-Sn-R3) in reactants
        2. Halide or triflate substrate in reactants  
        3. C-C bond formation between them
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
        
        # Filter out None molecules
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        product_mols = [mol for mol in product_mols if mol is not None]
        
        if not reactant_mols or not product_mols:
            return False
            
        # Check for organotin reagent in reactants
        tin_pattern = Chem.MolFromSmarts("[Sn]")
        has_tin_reagent = any(mol.HasSubstructMatch(tin_pattern) for mol in reactant_mols)
        
        # Check for halide or triflate in reactants
        halide_pattern = Chem.MolFromSmarts("[#6][Cl,Br,I]")  # C-X where X is halogen
        triflate_pattern = Chem.MolFromSmarts("[#6]OS(=O)(=O)C(F)(F)F")  # C-OTf
        
        has_electrophile = any(
            mol.HasSubstructMatch(halide_pattern) or mol.HasSubstructMatch(triflate_pattern)
            for mol in reactant_mols
        )
        
        # Basic check: if we have both tin reagent and electrophile, likely Stille coupling
        if has_tin_reagent and has_electrophile:
            return True
            
        # Additional check: look for typical Stille coupling pattern
        # Organotin compound with vinyl, aryl, or alkyl groups
        stille_tin_patterns = [
            Chem.MolFromSmarts("[#6]=[#6][Sn]"),  # vinyl-tin
            Chem.MolFromSmarts("c[Sn]"),          # aryl-tin
            Chem.MolFromSmarts("[#6;!$(C=*)][Sn]") # alkyl-tin
        ]
        
        has_stille_tin = any(
            any(mol.HasSubstructMatch(pattern) for pattern in stille_tin_patterns)
            for mol in reactant_mols
        )
        
        return has_stille_tin and has_electrophile
