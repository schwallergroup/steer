"""Generated evaluation code for: Late stage Williamson ether synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates routes for late-stage Williamson ether synthesis.
    Detects ether bond formation via nucleophilic substitution and rewards
    when it occurs at later stages of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late-stage (higher x values) is better
            if self.condition_type == "bool":
                return 1 if x >= self.target_depth else 0
            else:
                # Reward late-stage occurrence, penalize early-stage
                return max(0, 10 * (x - 0.2))
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by identifying:
        1. Formation of C-O-C ether bond
        2. Presence of alkyl halide or sulfonate leaving group
        3. Nucleophilic substitution pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for ether formation - look for new C-O-C bonds
            if not self._has_ether_formation(reactants, product):
                return False
            
            # Check for typical Williamson ether synthesis patterns
            return self._detect_williamson_pattern(reactants, product)
            
        except Exception:
            return False
    
    def _has_ether_formation(self, reactants, product) -> bool:
        """Check if new ether bonds (C-O-C) are formed"""
        # Count ether oxygens in product
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        product_ethers = len(product.GetSubstructMatches(ether_pattern))
        
        # Count ether oxygens in reactants
        reactant_ethers = sum(len(r.GetSubstructMatches(ether_pattern)) for r in reactants)
        
        return product_ethers > reactant_ethers
    
    def _detect_williamson_pattern(self, reactants, product) -> bool:
        """
        Detect typical Williamson ether synthesis patterns:
        - Alkoxide/phenoxide + alkyl halide/tosylate
        - Alcohol + alkyl halide under basic conditions
        """
        # Patterns for alkyl halides and sulfonates (leaving groups)
        alkyl_halide_patterns = [
            "[C][Cl]",  # alkyl chloride
            "[C][Br]",  # alkyl bromide
            "[C][I]",   # alkyl iodide
            "[C]OS(=O)(=O)[c]"  # tosylate
        ]
        
        # Patterns for nucleophiles
        nucleophile_patterns = [
            "[O-]",     # alkoxide anion
            "c[O-]",    # phenoxide anion
            "[OH]",     # alcohol (with base)
            "c[OH]"     # phenol (with base)
        ]
        
        # Check if reactants contain typical Williamson components
        has_leaving_group = False
        has_nucleophile = False
        
        for reactant in reactants:
            # Check for leaving groups
            for lg_pattern in alkyl_halide_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(lg_pattern)):
                    has_leaving_group = True
                    break
            
            # Check for nucleophiles
            for nuc_pattern in nucleophile_patterns:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(nuc_pattern)):
                    has_nucleophile = True
                    break
        
        # Also check for presence of base (common in Williamson ether synthesis)
        base_patterns = ["[Na+]", "[K+]", "[OH-]", "N(C)(C)C"]  # NaH, KOH, amines, etc.
        has_base = any(
            any(reactant.HasSubstructMatch(Chem.MolFromSmarts(base)) for base in base_patterns)
            for reactant in reactants
        )
        
        return has_leaving_group and (has_nucleophile or has_base)
