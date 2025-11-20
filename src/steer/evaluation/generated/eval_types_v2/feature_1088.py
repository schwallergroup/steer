"""Generated evaluation code for: Boc protecting group strategy for aniline"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocAnilineProtection(BaseScoring):
    """
    Evaluates the use of Boc protecting group strategy for aniline in synthesis routes.
    Checks for the presence of Boc-protected aniline intermediates and scores based on
    when this protection strategy is employed in the route.
    """
    
    def __init__(self, config: Dict):
        self.strategy_type = config.get("strategy", "temporary_protection")
        self.scoring_preference = config.get("scoring_preference", "early_stage")  # early_stage or late_stage
    
    def route_scoring(self, x) -> float:
        """
        Score based on when Boc protection of aniline occurs.
        Early stage protection is generally preferred for synthetic efficiency.
        """
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.scoring_preference == "early_stage":
            return 1 - x  # Earlier protection gets higher score
        else:
            return x  # Later protection gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc protection of aniline or uses Boc-protected aniline.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Check for Boc protection reaction (aniline + Boc reagent -> Boc-aniline)
            if self._is_boc_protection_reaction(reactants, products):
                return True
            
            # Check for use of Boc-protected aniline as reactant
            if self._contains_boc_aniline(reactants):
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_boc_protection_reaction(self, reactants: str, products: str) -> bool:
        """Check if this is a Boc protection reaction of aniline."""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for aniline in reactants
            aniline_pattern = Chem.MolFromSmarts("c1ccccc1N")
            has_aniline_reactant = any(mol.HasSubstructMatch(aniline_pattern) for mol in reactant_mols)
            
            # Check for Boc reagent in reactants (Boc2O or Boc-Cl patterns)
            boc_reagent_patterns = [
                Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl")  # Boc-Cl
            ]
            has_boc_reagent = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactant_mols)
                for pattern in boc_reagent_patterns if pattern
            )
            
            # Check for Boc-protected aniline in products
            boc_aniline_pattern = Chem.MolFromSmarts("c1ccccc1NC(=O)OC(C)(C)C")
            has_boc_aniline_product = any(mol.HasSubstructMatch(boc_aniline_pattern) for mol in product_mols)
            
            return has_aniline_reactant and has_boc_reagent and has_boc_aniline_product
            
        except Exception:
            return False
    
    def _contains_boc_aniline(self, reactants: str) -> bool:
        """Check if reactants contain Boc-protected aniline."""
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not all(reactant_mols):
                return False
            
            # Boc-protected aniline pattern
            boc_aniline_pattern = Chem.MolFromSmarts("c1ccccc1NC(=O)OC(C)(C)C")
            
            return any(mol.HasSubstructMatch(boc_aniline_pattern) for mol in reactant_mols)
            
        except Exception:
            return False
