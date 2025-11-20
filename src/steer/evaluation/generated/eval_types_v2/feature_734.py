"""Generated evaluation code for: Boc protecting group strategy for aniline"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocAnilineProtection(BaseScoring):
    """
    Evaluates Boc protecting group strategy for aniline functional groups.
    Checks if aniline is protected with Boc early in the synthesis and deprotected later.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.7)
        self.aniline_pattern = Chem.MolFromSmarts("c1ccccc1N")
        self.boc_aniline_pattern = Chem.MolFromSmarts("c1ccccc1NC(=O)OC(C)(C)C")
        self.boc_pattern = Chem.MolFromSmarts("NC(=O)OC(C)(C)C")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Reward protection happening early in synthesis (higher depth fraction)
            return 10 * (1 - abs(x - self.target_depth))

    def hit_condition(self, d):
        """Check if this reaction involves Boc protection of aniline"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for Boc protection: aniline in reactants -> Boc-aniline in product
            has_aniline_reactant = any(mol.HasSubstructMatch(self.aniline_pattern) 
                                     for mol in reactant_mols)
            has_boc_product = product_mol.HasSubstructMatch(self.boc_aniline_pattern)
            has_boc_reagent = any(mol.HasSubstructMatch(self.boc_pattern) 
                                for mol in reactant_mols)
            
            # Protection reaction: aniline + Boc reagent -> Boc-aniline
            if has_aniline_reactant and has_boc_product and has_boc_reagent:
                return True
                
            # Check for deprotection: Boc-aniline in reactants -> aniline in product
            has_boc_reactant = any(mol.HasSubstructMatch(self.boc_aniline_pattern) 
                                 for mol in reactant_mols)
            has_aniline_product = product_mol.HasSubstructMatch(self.aniline_pattern)
            
            # Deprotection reaction: Boc-aniline -> aniline
            if has_boc_reactant and has_aniline_product:
                return True
                
        except:
            return False
            
        return False
