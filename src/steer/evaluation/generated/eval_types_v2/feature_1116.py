"""Generated evaluation code for: Acetate protecting group for phenol chlorination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes for proper acetate protection of phenols before chlorination.
    
    Checks if phenols are protected with acetate groups prior to POCl3 chlorination reactions,
    which prevents unwanted side reactions during the chlorination step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "normalized")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
        # SMARTS patterns
        self.phenol_pattern = Chem.MolFromSmarts("[OH1][c]")  # Phenolic OH
        self.acetate_ester_pattern = Chem.MolFromSmarts("[c][O][C](=O)[CH3]")  # Phenyl acetate
        self.pocl3_chlorination_pattern = Chem.MolFromSmarts("[c][Cl]")  # Aromatic chloride
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            # Earlier protection is better (lower depth value)
            if x <= self.target_depth:
                return 10
            else:
                return max(0, 10 - 5 * (x - self.target_depth))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves acetate protection of phenol followed by chlorination.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if this is a protection step: phenol -> acetate ester
            protection_step = self._is_acetate_protection(reactant_mols, product_mol)
            
            # Check if this is a chlorination step with protected phenol
            chlorination_step = self._is_protected_chlorination(reactant_mols, product_mol)
            
            return protection_step or chlorination_step
            
        except Exception:
            return False
    
    def _is_acetate_protection(self, reactants, product):
        """Check if reaction converts phenol to acetate ester."""
        # Reactants should contain phenol
        has_phenol_reactant = any(mol.HasSubstructMatch(self.phenol_pattern) for mol in reactants)
        
        # Product should contain acetate ester
        has_acetate_product = product.HasSubstructMatch(self.acetate_ester_pattern)
        
        # Check that phenol OH is converted to acetate
        if has_phenol_reactant and has_acetate_product:
            # Verify acetylating agent present (acetic anhydride or acetyl chloride)
            acetylating_patterns = [
                Chem.MolFromSmarts("[CH3][C](=O)[O][C](=O)[CH3]"),  # Acetic anhydride
                Chem.MolFromSmarts("[CH3][C](=O)[Cl]")  # Acetyl chloride
            ]
            
            has_acetylating_agent = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactants)
                for pattern in acetylating_patterns
            )
            
            return has_acetylating_agent
        
        return False
    
    def _is_protected_chlorination(self, reactants, product):
        """Check if reaction is chlorination of acetate-protected phenol."""
        # Reactants should have acetate-protected phenol
        has_protected_reactant = any(mol.HasSubstructMatch(self.acetate_ester_pattern) for mol in reactants)
        
        # Product should have aromatic chloride and still be protected
        has_chlorinated_product = product.HasSubstructMatch(self.pocl3_chlorination_pattern)
        maintains_protection = product.HasSubstructMatch(self.acetate_ester_pattern)
        
        # Check for POCl3 or similar chlorinating agent
        chlorinating_patterns = [
            Chem.MolFromSmarts("P(=O)([Cl])([Cl])[Cl]"),  # POCl3
            Chem.MolFromSmarts("[Cl][Cl]"),  # Cl2
            Chem.MolFromSmarts("N([Cl])[C](=O)[C]")  # NCS-type
        ]
        
        has_chlorinating_agent = any(
            any(mol.HasSubstructMatch(pattern) for mol in reactants)
            for pattern in chlorinating_patterns if pattern
        )
        
        return (has_protected_reactant and has_chlorinated_product and 
                maintains_protection and has_chlorinating_agent)
