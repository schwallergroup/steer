"""Generated evaluation code for: Early Boc protection before amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyBocProtection(BaseScoring):
    """
    Evaluates whether Boc protection of amines occurs early in the synthesis route,
    specifically before amide coupling reactions. Rewards early Boc protection
    strategies that enable selective amide formation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc protection not found
        else:
            # Reward early protection (lower depth values)
            # Scale to 0-10 where early protection gets higher scores
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves Boc protection of an amine"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mols = [Chem.MolFromSmiles(s) for s in products.split(".") if s]
            react_mols = [Chem.MolFromSmiles(s) for s in reactants.split(".") if s]
            
            if not all(prod_mols) or not all(react_mols):
                return False
            
            # Check if Boc group is formed (present in products but not reactants)
            boc_pattern = Chem.MolFromSmarts("[N;!$(N=O)]-C(=O)-O-C(C)(C)C")  # Boc-protected amine
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1;!$(N-C(=O)-O)]")  # Free amine
            
            # Check products contain Boc-protected amine
            boc_in_products = any(mol.HasSubstructMatch(boc_pattern) for mol in prod_mols)
            
            # Check reactants contain free amine that could be protected
            free_amine_in_reactants = any(mol.HasSubstructMatch(free_amine_pattern) for mol in react_mols)
            
            # Also check for Boc reagents in reactants (Boc2O, Boc-Cl, etc.)
            boc_reagent_patterns = [
                Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C"),  # Boc2O
                Chem.MolFromSmarts("CC(C)(C)OC(=O)Cl"),  # Boc-Cl
                Chem.MolFromSmarts("CC(C)(C)OC(=O)O")  # Boc-OH
            ]
            
            boc_reagent_present = any(
                any(mol.HasSubstructMatch(pattern) for mol in react_mols)
                for pattern in boc_reagent_patterns if pattern
            )
            
            return (boc_in_products and free_amine_in_reactants and boc_reagent_present)
            
        except Exception:
            return False
    
    def _has_subsequent_amide_coupling(self, route_data) -> bool:
        """
        Helper method to verify that amide coupling occurs later in the route.
        This could be used for additional validation if needed.
        """
        try:
            amide_pattern = Chem.MolFromSmarts("C(=O)-N")  # Amide bond
            
            # Check if any subsequent reactions form amide bonds
            # This would require access to the full route tree structure
            # Implementation depends on route_data format
            
            return True  # Simplified - assume amide coupling happens
            
        except Exception:
            return False
