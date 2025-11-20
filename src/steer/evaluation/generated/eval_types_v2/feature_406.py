"""Generated evaluation code for: Late stage Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki cross-coupling reaction occurs at a late stage in the synthesis.
    Detects the formation of biaryl bonds through Suzuki coupling and scores based on timing.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage coupling is better (lower depth fraction = higher score)
            else:
                return x  # Early-stage coupling is better (higher depth fraction = higher score)
    
    def hit_condition(self, d):
        """Check if this reaction node represents a Suzuki coupling"""
        metadata = d.get("metadata", {})
        
        # Check if reaction smiles is available
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        try:
            prod_mol = Chem.MolFromSmiles(product)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r) is not None]
            
            if prod_mol is None or len(react_mols) < 2:
                return False
                
            # Check for Suzuki coupling pattern:
            # 1. Product should contain a biaryl bond (Ar-Ar)
            # 2. Reactants should contain organoborane and aryl halide patterns
            
            # Pattern for biaryl bond in product
            biaryl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
            
            if not prod_mol.HasSubstructMatch(biaryl_pattern):
                return False
                
            # Check reactants for Suzuki coupling partners
            has_organoborane = False
            has_aryl_halide = False
            
            # Organoborane patterns (aryl boronic acid, boronic ester, etc.)
            borane_patterns = [
                "c1ccccc1B(O)O",  # boronic acid
                "c1ccccc1B1OC(C)(C)C(C)(C)O1",  # pinacol boronic ester
                "c1ccccc1[B]",  # general boron
            ]
            
            # Aryl halide patterns
            halide_patterns = [
                "c1ccccc1Cl",
                "c1ccccc1Br", 
                "c1ccccc1I",
                "c1ccccc1F"
            ]
            
            for react_mol in react_mols:
                # Check for organoborane
                for pattern_smarts in borane_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and react_mol.HasSubstructMatch(pattern):
                        has_organoborane = True
                        break
                        
                # Check for aryl halide
                for pattern_smarts in halide_patterns:
                    pattern = Chem.MolFromSmarts(pattern_smarts)
                    if pattern and react_mol.HasSubstructMatch(pattern):
                        has_aryl_halide = True
                        break
                        
            return has_organoborane and has_aryl_halide
            
        except Exception:
            return False
