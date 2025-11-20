"""Generated evaluation code for: TBS protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBSProtectingGroupStrategy(BaseScoring):
    """
    Evaluates TBS (tert-butyldimethylsilyl) protecting group strategy for alcohols.
    Checks if TBS protection occurs at an appropriate depth in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.7)
        
        # SMARTS patterns for TBS group and alcohol
        self.tbs_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)C")  # TBS group
        self.alcohol_pattern = Chem.MolFromSmarts("[OH1]")  # Primary/secondary alcohol
        self.tbs_alcohol_pattern = Chem.MolFromSmarts("O[Si](C)(C)C(C)(C)C")  # TBS-protected alcohol

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # TBS protection doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Prefer TBS protection in early to mid-stage (depth 0.3-0.7)
            if x <= 0.3:
                return 5 + (x * 16.67)  # Score 5-10 for depths 0-0.3
            elif x <= 0.7:
                return 10 - ((x - 0.3) * 12.5)  # Score 10-5 for depths 0.3-0.7
            else:
                return max(0, 5 - ((x - 0.7) * 16.67))  # Score 5-0 for depths 0.7-1.0

    def hit_condition(self, d):
        """
        Check if this reaction involves TBS protection of an alcohol.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product has TBS-protected alcohol
            has_tbs_alcohol_product = product.HasSubstructMatch(self.tbs_alcohol_pattern)
            
            # Check if any reactant has free alcohol
            has_free_alcohol_reactant = any(r.HasSubstructMatch(self.alcohol_pattern) for r in reactants)
            
            # Check if any reactant contains TBS reagent (like TBSCl or TBSOTF)
            has_tbs_reagent = False
            for reactant in reactants:
                # Check for TBS-containing reagents
                if reactant.HasSubstructMatch(self.tbs_pattern):
                    has_tbs_reagent = True
                    break
                # Also check for common TBS reagent patterns
                tbs_cl_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)CCl")
                tbs_otf_pattern = Chem.MolFromSmarts("[Si](C)(C)C(C)(C)COS(=O)(=O)C(F)(F)F")
                if (reactant.HasSubstructMatch(tbs_cl_pattern) or 
                    reactant.HasSubstructMatch(tbs_otf_pattern)):
                    has_tbs_reagent = True
                    break
            
            # TBS protection occurs if:
            # 1. Product has TBS-protected alcohol AND
            # 2. Reactant has free alcohol AND  
            # 3. TBS reagent is present
            return (has_tbs_alcohol_product and 
                    has_free_alcohol_reactant and 
                    has_tbs_reagent)
                    
        except Exception:
            return False
