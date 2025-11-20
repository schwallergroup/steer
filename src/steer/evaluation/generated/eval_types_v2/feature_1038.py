"""Generated evaluation code for: Late stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Williamson ether synthesis.
    Checks if a Williamson ether synthesis (C-O bond formation between alkoxide and alkyl halide)
    occurs within the specified depth threshold from the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson synthesis doesn't occur
        elif x <= self.depth_threshold:
            return 1 - (x / 10)  # Earlier (lower depth) is better, scaled 0-1
        else:
            return 0  # Too early in the synthesis (beyond threshold)
    
    def hit_condition(self, d):
        """
        Detects Williamson ether synthesis by identifying:
        1. C-O bond formation in the product
        2. Presence of alkyl halide (C-X where X = Cl, Br, I) in reactants
        3. Presence of alkoxide or phenoxide nucleophile in reactants
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            product = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for alkyl halide pattern in reactants
            alkyl_halide_pattern = Chem.MolFromSmarts("[C][Cl,Br,I]")
            has_alkyl_halide = any(r.HasSubstructMatch(alkyl_halide_pattern) for r in reactants)
            
            if not has_alkyl_halide:
                return False
            
            # Check for alkoxide/phenoxide nucleophile (oxygen anion or hydroxyl)
            # Pattern for phenol or alcohol that could form alkoxide
            nucleophile_patterns = [
                Chem.MolFromSmarts("[O-]"),  # Alkoxide anion
                Chem.MolFromSmarts("c[OH]"),  # Phenol
                Chem.MolFromSmarts("[C][OH]")  # Alcohol
            ]
            
            has_nucleophile = any(
                any(r.HasSubstructMatch(pattern) for r in reactants) 
                for pattern in nucleophile_patterns
            )
            
            if not has_nucleophile:
                return False
            
            # Check for ether formation in product
            # Look for C-O-C pattern where oxygen is not in carbonyl
            ether_pattern = Chem.MolFromSmarts("[C][O][C]")
            has_ether_product = product.HasSubstructMatch(ether_pattern)
            
            # Additional check: ensure we're not matching carbonyl oxygens
            non_carbonyl_ether = Chem.MolFromSmarts("[C][O;!$(O=*)]")
            has_non_carbonyl_ether = product.HasSubstructMatch(non_carbonyl_ether)
            
            return has_ether_product and has_non_carbonyl_ether
            
        except Exception:
            return False
