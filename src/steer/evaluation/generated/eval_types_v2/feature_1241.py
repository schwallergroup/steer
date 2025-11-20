"""Generated evaluation code for: PMP acetal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PMPAcetalProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for PMP (p-methoxybenzaldehyde) acetal protecting group strategy.
    Checks for protection of 1,3-diols with PMP acetal and final deprotection to reveal target.
    """
    
    def __init__(self, config: Dict):
        self.require_final_deprotection = config["parameters"].get("final_deprotection", True)
        
        # PMP acetal pattern - p-methoxybenzaldehyde acetal protecting diol
        self.pmp_acetal_pattern = Chem.MolFromSmarts("[CH]1O[CH2][CH2]O1-c2ccc(OC)cc2")
        
        # Alternative patterns for PMP acetal
        self.pmp_acetal_alt = Chem.MolFromSmarts("COc1ccc(cc1)[CH]2O[CH2][CH2]O2")
        
        # p-methoxybenzaldehyde reactant pattern
        self.pmp_aldehyde = Chem.MolFromSmarts("COc1ccc(cc1)C=O")
        
        # 1,3-diol pattern
        self.diol_pattern = Chem.MolFromSmarts("[CH2:1][OH][CH2:2][OH]")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Earlier protection is better, score decreases with depth
            return max(0, 10 - (x * 10))

    def hit_condition(self, d) -> bool:
        """Check if this reaction involves PMP acetal protection strategy"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactants):
                return False
            
            # Check if product contains PMP acetal protecting group
            has_pmp_acetal = (product_mol.HasSubstructMatch(self.pmp_acetal_pattern) or 
                            product_mol.HasSubstructMatch(self.pmp_acetal_alt))
            
            if not has_pmp_acetal:
                return False
            
            # Check if reactants contain PMP aldehyde and diol
            has_pmp_aldehyde = any(r.HasSubstructMatch(self.pmp_aldehyde) for r in reactants)
            has_diol = any(r.HasSubstructMatch(self.diol_pattern) for r in reactants)
            
            # Protection reaction: diol + PMP aldehyde -> PMP acetal
            if has_pmp_aldehyde and has_diol:
                return True
            
            # Also check for deprotection if required
            if self.require_final_deprotection:
                return self._check_deprotection_context(d, product_mol, reactants)
                
            return False
            
        except Exception:
            return False
    
    def _check_deprotection_context(self, d, product_mol, reactants) -> bool:
        """Check if this is part of a protection-deprotection strategy"""
        # If any reactant has PMP acetal and product has diol, this is deprotection
        has_reactant_pmp = any(r.HasSubstructMatch(self.pmp_acetal_pattern) or 
                             r.HasSubstructMatch(self.pmp_acetal_alt) for r in reactants)
        has_product_diol = product_mol.HasSubstructMatch(self.diol_pattern)
        
        if has_reactant_pmp and has_product_diol:
            return True
            
        return False
