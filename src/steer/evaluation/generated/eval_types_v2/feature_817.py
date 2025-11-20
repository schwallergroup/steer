"""Generated evaluation code for: Early Sandmeyer halogenation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlySandmeyerHalogenation(BaseScoring):
    """
    Evaluates synthesis routes for early-stage Sandmeyer halogenation reactions.
    Detects Sandmeyer reactions (conversion of diazonium salts to halides) and
    rewards routes where this transformation occurs early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config.get("timing", "early")
    
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score (0-10).
        For early timing: lower depth (earlier) is better.
        """
        if x < 0:
            return 0  # Sandmeyer reaction doesn't occur
        
        if self.target_timing == "early":
            return (1 - x) * 10  # Early occurrence gets higher score
        else:
            return x * 10  # Late occurrence gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Detects Sandmeyer reaction by checking for diazonium salt conversion
        to halide pattern in the mapped reaction SMILES.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            # Parse product and reactants
            prod = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod or not all(reactants):
                return False
            
            # Check for Sandmeyer pattern: diazonium salt -> aryl halide
            # Diazonium pattern: aromatic N=[N+]
            diazonium_pattern = Chem.MolFromSmarts("[cH0:1][N:2]=[N+:3]")
            # Aryl halide pattern: aromatic carbon with halogen
            aryl_halide_pattern = Chem.MolFromSmarts("[c:1][F,Cl,Br,I]")
            
            # Check if product contains aryl halide
            if not prod.HasSubstructMatch(aryl_halide_pattern):
                return False
            
            # Check if any reactant contains diazonium salt
            has_diazonium = any(r.HasSubstructMatch(diazonium_pattern) for r in reactants)
            
            # Additional check: look for typical Sandmeyer reagents (CuCl, CuBr, KI, etc.)
            sandmeyer_reagents = [
                Chem.MolFromSmiles("[Cu+].[Cl-]"),  # CuCl
                Chem.MolFromSmiles("[Cu+].[Br-]"),  # CuBr
                Chem.MolFromSmiles("[K+].[I-]"),    # KI
                Chem.MolFromSmiles("[Cu+2].[Cl-].[Cl-]"),  # CuCl2
            ]
            
            has_sandmeyer_reagent = False
            for reagent in sandmeyer_reagents:
                if reagent:
                    for reactant in reactants:
                        if reactant.HasSubstructMatch(reagent):
                            has_sandmeyer_reagent = True
                            break
                    if has_sandmeyer_reagent:
                        break
            
            return has_diazonium and has_sandmeyer_reagent
            
        except Exception:
            return False
