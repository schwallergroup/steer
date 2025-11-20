"""Generated evaluation code for: Late stage ether formation via Williamson synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs at a late stage in the route.
    Detects ether formation from alkoxide/phenoxide and alkyl halide coupling.
    Rewards routes where this reaction happens closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.target_position = config.get("step_position", "final")
        self.timing_preference = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10), rewarding late-stage reactions"""
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing_preference == "late":
            # Higher score for reactions closer to final step (higher depth fraction)
            return 10 * x
        else:
            # Standard scoring
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents Williamson ether synthesis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._detect_williamson_ether_formation(product, reactants)
            
        except Exception:
            return False
    
    def _detect_williamson_ether_formation(self, product, reactants) -> bool:
        """
        Detect if the reaction represents Williamson ether synthesis:
        R-X + R'-O(-) -> R-O-R' + X(-)
        """
        # Check if product contains ether linkage
        ether_pattern = Chem.MolFromSmarts("[#6]-[#8]-[#6]")
        if not product.HasSubstructMatch(ether_pattern):
            return False
            
        # Look for characteristic reactant patterns
        has_alkyl_halide = False
        has_alkoxide = False
        
        # Patterns for alkyl/aryl halides
        halide_patterns = [
            "[#6]-[#9,#17,#35,#53]",  # C-halogen
            "[#6]-[#16]([#8])([#8])[#6]"  # tosylate-like leaving groups
        ]
        
        # Patterns for alkoxides/phenoxides
        alkoxide_patterns = [
            "[#6]-[#8-]",  # alkoxide anion
            "[#6]-[#8]",   # neutral alcohol (could be deprotonated in situ)
        ]
        
        for reactant in reactants:
            # Check for halide/leaving group
            for pattern_smarts in halide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkyl_halide = True
                    break
                    
            # Check for alkoxide/alcohol
            for pattern_smarts in alkoxide_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_alkoxide = True
                    break
        
        # Additional check: look for base (common in Williamson synthesis)
        base_patterns = [
            "[#11,#19]",  # Na, K (common bases)
            "[#8-]-[#1]",  # hydroxide
        ]
        
        has_base = False
        for reactant in reactants:
            for pattern_smarts in base_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and reactant.HasSubstructMatch(pattern):
                    has_base = True
                    break
        
        # Williamson synthesis requires alkyl halide + alkoxide
        return has_alkyl_halide and has_alkoxide
