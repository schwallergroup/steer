"""Generated evaluation code for: Early TBDMS deprotection before incompatible reagents"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBDMSDeprotectionTiming(BaseScoring):
    """
    Evaluates whether TBDMS deprotection occurs early before incompatible reagents.
    Returns high score if TBDMS is removed early, low score if removed late or not at all.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.timing = config["parameters"]["timing"]
        self.incompatible_reagents = config["parameters"]["incompatible_reagents"]
        
        # TBDMS pattern - tert-butyldimethylsilyl ether
        self.tbdms_pattern = "[Si](C)(C)C(C)(C)C"
        
    def route_scoring(self, x) -> float:
        """
        Score based on timing of TBDMS deprotection.
        Early deprotection (low x) gets high score, late gets low score.
        """
        if x < 0:
            return 0  # TBDMS deprotection doesn't happen
        
        # Convert depth fraction to score - earlier is better
        # x=0 (immediate) -> score=10, x=1 (very late) -> score=0
        return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves TBDMS deprotection.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactant_smiles, product_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)
            
            if not reactant_mol or not product_mol:
                return False
            
            # Create TBDMS substructure pattern
            tbdms_mol = Chem.MolFromSmarts(self.tbdms_pattern)
            if not tbdms_mol:
                return False
            
            # Check if TBDMS is present in reactant but absent in product
            has_tbdms_reactant = reactant_mol.HasSubstructMatch(tbdms_mol)
            has_tbdms_product = product_mol.HasSubstructMatch(tbdms_mol)
            
            # TBDMS deprotection: present in reactant, absent in product
            is_deprotection = has_tbdms_reactant and not has_tbdms_product
            
            # Also check if this involves incompatible reagents that would be problematic
            # if TBDMS was already removed (indicating this should happen early)
            involves_incompatible = self._involves_incompatible_reagents(mapped_rxn)
            
            return is_deprotection and involves_incompatible
            
        except Exception:
            return False
    
    def _involves_incompatible_reagents(self, mapped_rxn: str) -> bool:
        """
        Check if the reaction involves reagents incompatible with free hydroxyl groups.
        """
        rxn_lower = mapped_rxn.lower()
        
        # Check for SOCl2 (thionyl chloride)
        if "socl2" in self.incompatible_reagents:
            if "s" in rxn_lower and "cl" in rxn_lower:
                return True
        
        # Check for Grignard reagents (Mg with alkyl/aryl groups)
        if "grignard" in [r.lower() for r in self.incompatible_reagents]:
            if "mg" in rxn_lower or "[mg" in rxn_lower:
                return True
        
        # Additional check for common incompatible reagent patterns
        incompatible_patterns = [
            "socl2", "pocl3", "mgbr", "mgi", "mgcl"
        ]
        
        return any(pattern in rxn_lower for pattern in incompatible_patterns)
