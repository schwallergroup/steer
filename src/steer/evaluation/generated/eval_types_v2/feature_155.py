"""Generated evaluation code for: Benzyl protecting group strategy for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates benzyl protecting group strategy for phenol groups.
    Checks for benzyl ether formation (protection) and hydrogenolytic deprotection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is generally better (closer to 0)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """Check if this reaction involves benzyl protection/deprotection of phenol"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
                
            # Check for benzyl ether protection (phenol + benzyl halide -> benzyl ether)
            if self._is_protection_reaction(prod_mol, react_mols):
                return True
                
            # Check for hydrogenolytic deprotection (benzyl ether -> phenol + toluene)
            if self._is_deprotection_reaction(prod_mol, react_mols):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_protection_reaction(self, product, reactants):
        """Check if reaction forms benzyl ether from phenol"""
        # Benzyl ether pattern: phenolic oxygen connected to benzyl carbon
        benzyl_ether_pattern = Chem.MolFromSmarts("[OH1]-[CH2]-c1ccccc1")
        phenol_pattern = Chem.MolFromSmarts("c-[OH1]")
        benzyl_halide_pattern = Chem.MolFromSmarts("[CH2]([Cl,Br,I])-c1ccccc1")
        benzyl_alcohol_pattern = Chem.MolFromSmarts("[CH2]([OH1])-c1ccccc1")
        
        if not benzyl_ether_pattern or not phenol_pattern:
            return False
            
        # Product should contain benzyl ether
        has_benzyl_ether = product.HasSubstructMatch(benzyl_ether_pattern)
        
        # Reactants should contain phenol and benzyl source
        has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactants)
        has_benzyl_source = any(
            mol.HasSubstructMatch(benzyl_halide_pattern) or 
            mol.HasSubstructMatch(benzyl_alcohol_pattern) 
            for mol in reactants
        )
        
        return has_benzyl_ether and has_phenol and has_benzyl_source
    
    def _is_deprotection_reaction(self, product, reactants):
        """Check if reaction cleaves benzyl ether to form phenol (hydrogenation)"""
        benzyl_ether_pattern = Chem.MolFromSmarts("c-[OH1]-[CH2]-c1ccccc1")
        phenol_pattern = Chem.MolFromSmarts("c-[OH1]")
        toluene_pattern = Chem.MolFromSmarts("c1ccc(C)cc1")
        
        if not benzyl_ether_pattern or not phenol_pattern:
            return False
            
        # Reactants should contain benzyl ether
        has_benzyl_ether = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in reactants)
        
        # Products should contain phenol and potentially toluene
        has_phenol = product.HasSubstructMatch(phenol_pattern)
        
        # Check if toluene is among products (split products by '.')
        product_smiles = Chem.MolToSmiles(product)
        if '.' in product_smiles:
            product_parts = [Chem.MolFromSmiles(p.strip()) for p in product_smiles.split('.')]
            has_toluene = any(mol and mol.HasSubstructMatch(toluene_pattern) for mol in product_parts if mol)
        else:
            has_toluene = product.HasSubstructMatch(toluene_pattern)
        
        return has_benzyl_ether and has_phenol
