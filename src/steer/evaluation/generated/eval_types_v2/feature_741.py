"""Generated evaluation code for: Boc protecting group for pyridine amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocPyridineAmineProtection(BaseScoring):
    """
    Evaluates whether Boc protection of a pyridine amine occurs at the appropriate timing.
    Checks for Boc protection of primary amines on pyridine rings, with preference for
    late-stage deprotection.
    """
    
    def __init__(self, config: Dict):
        self.deprotection_timing = config.get("deprotection_timing", "late")
        self.condition_type = config.get("target_depth", {}).get("type", "value")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.deprotection_timing == "late":
            # Reward late-stage deprotection (higher depth values)
            return min(10, x * 10)
        else:
            # For early deprotection, invert the preference
            return min(10, (1 - x) * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of pyridine amine"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
            
            # Check for Boc protection: pyridine amine -> Boc-protected pyridine amine
            return self._is_boc_protection_rxn(prod_mol, react_mols) or \
                   self._is_boc_deprotection_rxn(prod_mol, react_mols)
                   
        except Exception:
            return False
    
    def _is_boc_protection_rxn(self, product, reactants) -> bool:
        """Check if reaction is Boc protection of pyridine amine"""
        # Boc-protected pyridine amine pattern
        boc_pyridine_pattern = Chem.MolFromSmarts("[cH1]1[cH1][cH1][cH1][nH0][cH1]1-[NH1]-C(=O)-O-C(C)(C)C")
        # Free pyridine amine pattern  
        free_pyridine_pattern = Chem.MolFromSmarts("[cH1]1[cH1][cH1][cH1][nH0][cH1]1-[NH2]")
        
        if not boc_pyridine_pattern or not free_pyridine_pattern:
            return False
            
        # Product should have Boc-protected amine
        has_boc_product = product.HasSubstructMatch(boc_pyridine_pattern)
        
        # At least one reactant should have free pyridine amine
        has_free_reactant = any(mol.HasSubstructMatch(free_pyridine_pattern) for mol in reactants)
        
        # Check for Boc reagent in reactants (tert-butyl dicarbonate or similar)
        boc_reagent_pattern = Chem.MolFromSmarts("C(=O)-O-C(=O)-O-C(C)(C)C")
        has_boc_reagent = any(mol.HasSubstructMatch(boc_reagent_pattern) for mol in reactants)
        
        return has_boc_product and has_free_reactant and has_boc_reagent
    
    def _is_boc_deprotection_rxn(self, product, reactants) -> bool:
        """Check if reaction is Boc deprotection of pyridine amine"""
        # Boc-protected pyridine amine pattern
        boc_pyridine_pattern = Chem.MolFromSmarts("[cH1]1[cH1][cH1][cH1][nH0][cH1]1-[NH1]-C(=O)-O-C(C)(C)C")
        # Free pyridine amine pattern
        free_pyridine_pattern = Chem.MolFromSmarts("[cH1]1[cH1][cH1][cH1][nH0][cH1]1-[NH2]")
        
        if not boc_pyridine_pattern or not free_pyridine_pattern:
            return False
            
        # Product should have free pyridine amine
        has_free_product = product.HasSubstructMatch(free_pyridine_pattern)
        
        # At least one reactant should have Boc-protected amine
        has_boc_reactant = any(mol.HasSubstructMatch(boc_pyridine_pattern) for mol in reactants)
        
        return has_free_product and has_boc_reactant
