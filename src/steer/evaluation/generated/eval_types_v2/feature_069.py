"""Generated evaluation code for: Cyclic carbamate protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclicCarbamateStrategy(BaseScoring):
    """
    Evaluates routes based on the use of cyclic carbamate protecting groups.
    
    This class checks if a synthesis route employs a 1,3-oxazinan-2-one cyclic carbamate
    strategy to simultaneously protect both an amine and hydroxyl group, which is useful
    for amine-diol substrates.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
        
        # SMARTS pattern for 1,3-oxazinan-2-one cyclic carbamate core
        self.cyclic_carbamate_pattern = "O1CCNC(=O)O1"
        # Alternative patterns for related cyclic carbamates
        self.alt_patterns = [
            "O1C[CH2]NC(=O)O1",  # 5-membered oxazolidinone
            "O1CC[NH]C(=O)O1",   # More flexible matching
            "[O]1[CH2][CH2][NH][C](=[O])[O]1"  # Explicit pattern
        ]

    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)."""
        if x < 0:
            return 0  # Strategy not used
        
        if self.condition_type == "bool":
            return 10  # Found the strategy
        else:
            # Earlier use of protecting group strategy is generally better
            return max(0, 10 * (1 - abs(x - self.target_depth)))

    def hit_condition(self, d) -> bool:
        """
        Check if a reaction involves cyclic carbamate protecting group chemistry.
        
        This looks for:
        1. Formation of cyclic carbamate (protection step)
        2. Cleavage of cyclic carbamate (deprotection step)
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
                
            # Check for cyclic carbamate formation (protection)
            if self._is_protection_step(reactant_mols, product_mols):
                return True
                
            # Check for cyclic carbamate cleavage (deprotection)  
            if self._is_deprotection_step(reactant_mols, product_mols):
                return True
                
        except Exception:
            return False
            
        return False

    def _is_protection_step(self, reactants, products) -> bool:
        """Check if this is a cyclic carbamate protection step."""
        # Look for cyclic carbamate in products but not reactants
        reactant_has_carbamate = any(self._has_cyclic_carbamate(mol) for mol in reactants)
        product_has_carbamate = any(self._has_cyclic_carbamate(mol) for mol in products)
        
        return product_has_carbamate and not reactant_has_carbamate

    def _is_deprotection_step(self, reactants, products) -> bool:
        """Check if this is a cyclic carbamate deprotection step."""
        # Look for cyclic carbamate in reactants but not products
        reactant_has_carbamate = any(self._has_cyclic_carbamate(mol) for mol in reactants)
        product_has_carbamate = any(self._has_cyclic_carbamate(mol) for mol in products)
        
        return reactant_has_carbamate and not product_has_carbamate

    def _has_cyclic_carbamate(self, mol) -> bool:
        """Check if molecule contains a cyclic carbamate motif."""
        if not mol:
            return False
            
        # Check main pattern
        pattern_mol = Chem.MolFromSmarts(self.cyclic_carbamate_pattern)
        if pattern_mol and mol.HasSubstructMatch(pattern_mol):
            return True
            
        # Check alternative patterns
        for pattern_smarts in self.alt_patterns:
            pattern_mol = Chem.MolFromSmarts(pattern_smarts)
            if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                return True
                
        return False
