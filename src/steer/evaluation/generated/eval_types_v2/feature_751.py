"""Generated evaluation code for: Benzyl protecting group strategy for phenol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylPhenolProtection(BaseScoring):
    """
    Evaluates benzyl protecting group strategy for phenols.
    Checks if benzyl ether protection is used on phenols and later removed via hydrogenolysis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "linear")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better (allows more synthetic manipulations)
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves benzyl protection of phenol"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for benzyl protection: phenol + benzyl halide -> benzyl ether
            if self._is_benzyl_protection(reactants, products):
                return True
                
            # Check for hydrogenolysis deprotection: benzyl ether -> phenol + toluene
            if self._is_hydrogenolysis_deprotection(reactants, products):
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_benzyl_protection(self, reactants: str, products: str) -> bool:
        """Check if reaction is benzyl protection of phenol"""
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Phenol pattern
            phenol_pattern = Chem.MolFromSmarts("c[OH1]")
            # Benzyl halide pattern (Br, Cl, I)
            benzyl_halide_pattern = Chem.MolFromSmarts("c1ccccc1C[Br,Cl,I]")
            # Benzyl ether pattern
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
            
            # Check reactants contain phenol and benzyl halide
            has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in reactant_mols)
            has_benzyl_halide = any(mol.HasSubstructMatch(benzyl_halide_pattern) for mol in reactant_mols)
            
            # Check products contain benzyl ether
            has_benzyl_ether = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in product_mols)
            
            return has_phenol and has_benzyl_halide and has_benzyl_ether
            
        except Exception:
            return False
    
    def _is_hydrogenolysis_deprotection(self, reactants: str, products: str) -> bool:
        """Check if reaction is hydrogenolysis deprotection of benzyl ether"""
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Benzyl ether pattern
            benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
            # Phenol pattern
            phenol_pattern = Chem.MolFromSmarts("c[OH1]")
            # Toluene pattern (product of benzyl group reduction)
            toluene_pattern = Chem.MolFromSmarts("c1ccccc1C")
            
            # Check reactants contain benzyl ether
            has_benzyl_ether = any(mol.HasSubstructMatch(benzyl_ether_pattern) for mol in reactant_mols)
            
            # Check products contain phenol and toluene
            has_phenol = any(mol.HasSubstructMatch(phenol_pattern) for mol in product_mols)
            has_toluene = any(mol.HasSubstructMatch(toluene_pattern) for mol in product_mols)
            
            return has_benzyl_ether and has_phenol and has_toluene
            
        except Exception:
            return False
