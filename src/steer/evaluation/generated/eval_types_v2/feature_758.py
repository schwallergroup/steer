"""Generated evaluation code for: Late stage Suzuki coupling for fragment union"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs at a late stage in the synthesis
    to unite two major fragments in a convergent strategy.
    """
    
    def __init__(self, config: Dict):
        self.timing_threshold = config.get("timing_threshold", 0.2)  # Within 20% of route end
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        else:
            # Score based on how late the reaction occurs (closer to 0 is better)
            if x <= self.timing_threshold:
                return 10  # Perfect score for very late stage
            else:
                # Linear decay from 10 to 0 as depth increases
                return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling between two significant fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or len(reactants) < 2:
                return False
                
            # Check for Suzuki coupling pattern: organoborane + organohalide -> C-C bond
            has_borane = False
            has_halide = False
            
            # Patterns for Suzuki coupling reactants
            borane_pattern = Chem.MolFromSmarts("[C]-B(-[OH])(-[OH])")  # Boronic acid
            borane_pattern2 = Chem.MolFromSmarts("[C]-B1-O[C]CO1")      # Boronic ester
            halide_pattern = Chem.MolFromSmarts("[C]-[Br,I,Cl]")        # Aryl/vinyl halide
            
            # Check each reactant
            for reactant in reactants:
                if reactant and reactant.GetNumAtoms() > 5:  # Filter out small molecules/catalysts
                    if (reactant.HasSubstructMatch(borane_pattern) or 
                        reactant.HasSubstructMatch(borane_pattern2)):
                        has_borane = True
                    elif reactant.HasSubstructMatch(halide_pattern):
                        has_halide = True
            
            # Must have both coupling partners and they should be substantial fragments
            if has_borane and has_halide:
                # Check that we have two major fragments (not tiny coupling partners)
                major_fragments = [r for r in reactants if r and r.GetNumAtoms() > 8]
                return len(major_fragments) >= 2
                
            return False
            
        except Exception:
            return False
