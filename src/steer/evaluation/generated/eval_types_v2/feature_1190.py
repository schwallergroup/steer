"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes based on the use of benzyl ether protecting groups for alcohols.
    Checks if a benzyl ether protection strategy is employed at the specified depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "exact")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not found
        
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0  # Just check if protection is used
        else:
            # Earlier protection is generally better (lower depth)
            if x < 0:
                return 0
            return max(0, 10 - x * 2)  # Scale depth to 0-10 score
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzyl ether protection of an alcohol.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            return self._detect_benzyl_ether_protection(reactants, products)
            
        except Exception:
            return False
    
    def _detect_benzyl_ether_protection(self, reactants, products):
        """
        Detect if benzyl ether protection of alcohol occurs in this reaction.
        """
        # Benzyl ether pattern: benzyl group connected to oxygen
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1-O")
        benzyl_bromide_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1Br")
        benzyl_chloride_pattern = Chem.MolFromSmarts("[CH2]c1ccccc1Cl")
        alcohol_pattern = Chem.MolFromSmarts("[CH,CH2,CH3]-[OH]")
        
        if not all([benzyl_ether_pattern, benzyl_bromide_pattern, benzyl_chloride_pattern, alcohol_pattern]):
            return False
        
        # Check if products contain benzyl ether that wasn't in reactants
        reactant_has_benzyl_ether = any(
            mol.HasSubstructMatch(benzyl_ether_pattern) for mol in reactants
        )
        
        product_has_benzyl_ether = any(
            mol.HasSubstructMatch(benzyl_ether_pattern) for mol in products
        )
        
        # Check if reactants contain alcohol and benzyl halide
        has_alcohol = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactants)
        has_benzyl_halide = any(
            mol.HasSubstructMatch(benzyl_bromide_pattern) or 
            mol.HasSubstructMatch(benzyl_chloride_pattern)
            for mol in reactants
        )
        
        # Protection: alcohol + benzyl halide -> benzyl ether (formation)
        protection_occurring = (
            not reactant_has_benzyl_ether and 
            product_has_benzyl_ether and 
            has_alcohol and 
            has_benzyl_halide
        )
        
        return protection_occurring
