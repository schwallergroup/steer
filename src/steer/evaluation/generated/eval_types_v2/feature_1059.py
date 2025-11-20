"""Generated evaluation code for: Boc protecting group for amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of Boc protecting group strategy on amines.
    Checks if Boc protection/deprotection reactions occur and at what depth.
    Earlier use of Boc protection is generally preferred for route efficiency.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Boc protection found
        else:
            # Earlier Boc protection is better (lower depth fraction)
            # Score ranges from 0-10, with early protection scoring higher
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d):
        """Check if this reaction involves Boc protection or deprotection of an amine."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Check for Boc protection (amine + Boc reagent -> Boc-protected amine)
        if self._is_boc_protection(reactants_smiles, products_smiles):
            return True
            
        # Check for Boc deprotection (Boc-protected amine -> amine)
        if self._is_boc_deprotection(reactants_smiles, products_smiles):
            return True
            
        return False
    
    def _is_boc_protection(self, reactants_smiles, products_smiles):
        """Check if reaction is Boc protection of an amine."""
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for presence of free amine in reactants
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1][!C(=O)O]")
            has_free_amine_reactant = any(mol.HasSubstructMatch(free_amine_pattern) for mol in reactant_mols)
            
            # Check for Boc reagent in reactants (Boc2O or Boc-Cl type reagents)
            boc_reagent_patterns = [
                Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl")  # Boc-Cl
            ]
            has_boc_reagent = any(
                any(mol.HasSubstructMatch(pattern) for pattern in boc_reagent_patterns)
                for mol in reactant_mols
            )
            
            # Check for Boc-protected amine in products
            boc_protected_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            has_boc_product = any(mol.HasSubstructMatch(boc_protected_pattern) for mol in product_mols)
            
            return has_free_amine_reactant and has_boc_reagent and has_boc_product
            
        except Exception:
            return False
    
    def _is_boc_deprotection(self, reactants_smiles, products_smiles):
        """Check if reaction is Boc deprotection to reveal an amine."""
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for Boc-protected amine in reactants
            boc_protected_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
            has_boc_reactant = any(mol.HasSubstructMatch(boc_protected_pattern) for mol in reactant_mols)
            
            # Check for free amine in products
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1][!C(=O)O]")
            has_free_amine_product = any(mol.HasSubstructMatch(free_amine_pattern) for mol in product_mols)
            
            # Check for typical Boc deprotection byproducts (CO2, isobutene)
            deprotection_byproducts = [
                Chem.MolFromSmiles("O=C=O"),  # CO2
                Chem.MolFromSmiles("C=C(C)C")  # isobutene
            ]
            has_deprotection_byproducts = any(
                any(Chem.MolToSmiles(mol) == Chem.MolToSmiles(byproduct) for byproduct in deprotection_byproducts)
                for mol in product_mols
            )
            
            return has_boc_reactant and has_free_amine_product
            
        except Exception:
            return False
