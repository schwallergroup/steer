"""Generated evaluation code for: Boc protection strategy for chemoselectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectionStrategy(BaseScoring):
    """
    Evaluates the use of Boc protection strategy for chemoselectivity.
    Checks if Boc protection/deprotection is used to protect secondary amines
    and prevent competing N-alkylation reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Boc protection strategy found
        else:
            # Earlier use of Boc protection is better for chemoselectivity
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc protection or deprotection
        of a secondary amine for chemoselectivity purposes.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".") if smi]
        products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".") if smi]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        if not reactants or not products:
            return False
            
        # Check for Boc protection (amine -> Boc-protected amine)
        if self._is_boc_protection(reactants, products):
            return True
            
        # Check for Boc deprotection (Boc-protected amine -> amine)
        if self._is_boc_deprotection(reactants, products):
            return True
            
        return False
    
    def _is_boc_protection(self, reactants, products) -> bool:
        """Check if reaction is Boc protection of secondary amine."""
        # Boc reagent pattern (Boc2O or Boc-Cl)
        boc_reagent_patterns = [
            "[CH3][CH3][CH3]OC(=O)OC(=O)O[CH3]([CH3])[CH3]",  # Boc2O
            "[CH3][CH3][CH3]OC(=O)Cl"  # Boc-Cl
        ]
        
        # Secondary amine pattern
        secondary_amine_pattern = "[NH1]([CH2,CH3,c])[CH2,CH3,c]"
        
        # Boc-protected secondary amine pattern
        boc_protected_pattern = "[NH0]([CH2,CH3,c])([CH2,CH3,c])C(=O)O[CH3]([CH3])[CH3]"
        
        # Check if reactants contain Boc reagent and secondary amine
        has_boc_reagent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in boc_reagent_patterns)
            for mol in reactants
        )
        
        has_secondary_amine = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(secondary_amine_pattern))
            for mol in reactants
        )
        
        # Check if products contain Boc-protected amine
        has_boc_protected = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(boc_protected_pattern))
            for mol in products
        )
        
        return has_boc_reagent and has_secondary_amine and has_boc_protected
    
    def _is_boc_deprotection(self, reactants, products) -> bool:
        """Check if reaction is Boc deprotection to reveal secondary amine."""
        # Boc-protected secondary amine pattern
        boc_protected_pattern = "[NH0]([CH2,CH3,c])([CH2,CH3,c])C(=O)O[CH3]([CH3])[CH3]"
        
        # Secondary amine pattern
        secondary_amine_pattern = "[NH1]([CH2,CH3,c])[CH2,CH3,c]"
        
        # Common deprotection conditions (TFA, HCl)
        acid_patterns = [
            "FC(F)(F)C(=O)O",  # TFA
            "Cl",  # HCl
            "[H+]"  # Generic acid
        ]
        
        # Check if reactants contain Boc-protected amine
        has_boc_protected = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(boc_protected_pattern))
            for mol in reactants
        )
        
        # Check if products contain secondary amine
        has_secondary_amine = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(secondary_amine_pattern))
            for mol in products
        )
        
        # Check for acid conditions (optional, as acids might not be explicit)
        has_acid = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in acid_patterns if Chem.MolFromSmarts(pattern) is not None)
            for mol in reactants
        )
        
        return has_boc_protected and has_secondary_amine
