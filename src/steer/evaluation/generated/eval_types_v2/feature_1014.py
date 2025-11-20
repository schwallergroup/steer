"""Generated evaluation code for: TBS protection of primary alcohol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBSProtectionStrategy(BaseScoring):
    """
    Evaluates whether TBS (tert-butyldimethylsilyl) protection of primary alcohols 
    occurs in the synthesis route at the appropriate depth.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        self.present = config["parameters"]["present"]
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.present:
                return 1 if x >= 0 else 0  # Positive if TBS protection found
            else:
                return 1 if x < 0 else 0   # Positive if TBS protection not found
        else:
            if x < 0:
                return 0 if self.present else 1
            return 1 - abs(x - self.target_depth) / 10  # Score based on depth proximity

    def hit_condition(self, d):
        """Check if this reaction involves TBS protection of primary alcohol"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check for TBS protection reaction
            return self._is_tbs_protection_reaction(reactants, products)
            
        except:
            return False

    def _is_tbs_protection_reaction(self, reactants, products):
        """Detect TBS protection of primary alcohols"""
        
        # TBS reagent patterns (common TBS protecting reagents)
        tbs_reagent_patterns = [
            "[Si](C)(C)C(C)(C)C",  # TBS group core
            "CC(C)(C)[Si](C)(C)Cl",  # TBSCl
            "CC(C)(C)[Si](C)(C)O[Si](C)(C)C(C)(C)C",  # TBS2O
        ]
        
        # Primary alcohol pattern
        primary_alcohol_pattern = "[CH2][OH]"
        
        # TBS-protected primary alcohol pattern
        tbs_protected_pattern = "[CH2]O[Si](C)(C)C(C)(C)C"
        
        # Check if reactants contain primary alcohol and TBS reagent
        has_primary_alcohol = False
        has_tbs_reagent = False
        
        for reactant in reactants:
            # Check for primary alcohol
            if reactant.HasSubstructMatch(Chem.MolFromSmarts(primary_alcohol_pattern)):
                has_primary_alcohol = True
            
            # Check for TBS reagent
            for pattern in tbs_reagent_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_tbs_reagent = True
                        break
                except:
                    continue
        
        # Check if products contain TBS-protected alcohol
        has_tbs_protected = False
        for product in products:
            try:
                if product.HasSubstructMatch(Chem.MolFromSmarts(tbs_protected_pattern)):
                    has_tbs_protected = True
                    break
            except:
                continue
        
        # Alternative check: look for the transformation pattern
        # Primary alcohol in reactants -> TBS-protected alcohol in products
        if has_primary_alcohol and has_tbs_protected and not self._has_tbs_protected_in_reactants(reactants):
            return True
            
        # Direct reagent-based detection
        return has_primary_alcohol and has_tbs_reagent and has_tbs_protected

    def _has_tbs_protected_in_reactants(self, reactants):
        """Check if reactants already contain TBS-protected alcohols"""
        tbs_protected_pattern = "[CH2]O[Si](C)(C)C(C)(C)C"
        for reactant in reactants:
            try:
                if reactant.HasSubstructMatch(Chem.MolFromSmarts(tbs_protected_pattern)):
                    return True
            except:
                continue
        return False
