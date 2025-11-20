"""Generated evaluation code for: Late stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonSynthesis(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs at late stage (shallow depth).
    Detects C-O-C ether bond formation from alkoxide and alkyl halide reactants.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson synthesis doesn't occur
        
        # Late stage is better - penalize deeper reactions
        depth_fraction = x
        if depth_fraction <= self.depth_threshold / 10.0:  # Within threshold
            return 1 - depth_fraction  # Perfect score for depth 0, decreasing
        else:
            return max(0, 1 - depth_fraction * 2)  # Penalty for being too deep
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by checking for:
        1. Product has C-O-C ether linkage
        2. Reactants contain alkyl halide (C-X where X = Br, I, Cl) 
        3. Reactants contain alkoxide precursor (alcohol or alkoxide)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".") if r]
            
            if not product or not reactants:
                return False
            
            # Check if product contains ether (C-O-C, not C=O)
            ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
            if not product.HasSubstructMatch(ether_pattern):
                return False
            
            # Check reactants for Williamson synthesis components
            has_alkyl_halide = False
            has_alcohol_or_alkoxide = False
            
            # Patterns for alkyl halides (primary or secondary preferred)
            alkyl_halide_patterns = [
                "[C][Br]",  # alkyl bromide
                "[C][I]",   # alkyl iodide  
                "[C][Cl]"   # alkyl chloride
            ]
            
            # Patterns for alcohol/alkoxide
            alcohol_patterns = [
                "[C][OH]",     # alcohol
                "[C][O-]",     # alkoxide anion
                "[C][O][Na]",  # sodium alkoxide
                "[C][O][K]"    # potassium alkoxide
            ]
            
            for reactant in reactants:
                # Check for alkyl halide
                for pattern in alkyl_halide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alkyl_halide = True
                        break
                
                # Check for alcohol/alkoxide
                for pattern in alcohol_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alcohol_or_alkoxide = True
                        break
            
            return has_alkyl_halide and has_alcohol_or_alkoxide
            
        except Exception:
            return False
