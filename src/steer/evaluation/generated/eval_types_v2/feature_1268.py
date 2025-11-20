"""Generated evaluation code for: Benzyl ether protecting group strategy for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtection(BaseScoring):
    """
    Evaluates synthesis routes for benzyl ether protecting group strategy on phenols.
    Checks for benzyl protection of phenols and subsequent hydrogenolysis removal.
    """
    
    def __init__(self, config: Dict):
        self.strategy_type = config.get("strategy_type", "protection")  # "protection" or "deprotection"
        self.require_both = config.get("require_both", False)  # Require both protection and deprotection
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            return 1 - x  # Earlier use of strategy is better
    
    def hit_condition(self, d):
        """Check if this reaction involves benzyl ether protection/deprotection of phenol"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Check for benzyl protection (phenol + benzyl halide -> benzyl ether)
            if self._is_benzyl_protection(reactants, products):
                return True
                
            # Check for benzyl deprotection (benzyl ether -> phenol via hydrogenolysis)
            if self._is_benzyl_deprotection(reactants, products):
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_benzyl_protection(self, reactants, products):
        """Check if reaction is phenol benzylation"""
        # Phenol pattern (aromatic OH)
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        # Benzyl halide pattern
        benzyl_halide_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1")
        # Benzyl ether pattern
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1-O-[c]")
        
        # Check if reactants contain phenol and benzyl halide
        has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactants)
        has_benzyl_halide = any(mol.HasSubstructMatch(benzyl_halide_pattern) for mol in reactants)
        
        # Check if products contain benzyl ether
        has_benzyl_ether = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in products)
        
        return has_phenol and has_benzyl_halide and has_benzyl_ether
    
    def _is_benzyl_deprotection(self, reactants, products):
        """Check if reaction is benzyl ether hydrogenolysis"""
        # Benzyl ether pattern
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2][c]1[cH][cH][cH][cH][cH]1-O-[c]")
        # Phenol pattern
        phenol_pattern = Chem.MolFromSmarts("[OH1][c]")
        # Toluene pattern (product of benzyl removal)
        toluene_pattern = Chem.MolFromSmarts("[CH3][c]1[cH][cH][cH][cH][cH]1")
        
        # Check if reactants contain benzyl ether
        has_benzyl_ether = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in reactants)
        
        # Check if products contain phenol and toluene (or similar benzyl fragment)
        has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in products)
        has_toluene = any(mol.HasSubstructMatch(toluene_pattern) for mol in products)
        
        return has_benzyl_ether and has_phenol and has_toluene
