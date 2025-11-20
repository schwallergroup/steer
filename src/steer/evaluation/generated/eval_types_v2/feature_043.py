"""Generated evaluation code for: Phenylsulfonyl protecting group strategy for pyrrole nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhenylsulfonylPyrroleProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of phenylsulfonyl protecting groups
    on pyrrole nitrogen atoms. Returns higher scores for routes that employ
    this protecting group strategy at appropriate depths.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns
        self.phenylsulfonyl_pyrrole_pattern = "[c:1]1[c:2][c:3][c:4][c:5][c:6]1-[S:7](=[O:8])(=[O:9])-[n:10]1[c:11][c:12][c:13][c:14]1"
        self.free_pyrrole_pattern = "[nH:1]1[c:2][c:3][c:4][c:5]1"
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            # Earlier protection is generally better for synthetic planning
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves phenylsulfonyl protection of pyrrole nitrogen
        or deprotection under harsh conditions.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for protection reaction: free pyrrole + phenylsulfonyl reagent -> protected pyrrole
            protection_reaction = self._is_protection_reaction(reactants, products)
            
            # Check for deprotection reaction: protected pyrrole -> free pyrrole
            deprotection_reaction = self._is_deprotection_reaction(reactants, products)
            
            return protection_reaction or deprotection_reaction
            
        except Exception:
            return False
    
    def _is_protection_reaction(self, reactants, products) -> bool:
        """Check if reaction converts free pyrrole to phenylsulfonyl-protected pyrrole"""
        # Look for free pyrrole in reactants
        has_free_pyrrole_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_pyrrole_pattern))
            for mol in reactants
        )
        
        # Look for phenylsulfonyl reagent in reactants (e.g., phenylsulfonyl chloride)
        phenylsulfonyl_reagent_pattern = "c1ccccc1-S(=O)(=O)-Cl"
        has_phenylsulfonyl_reagent = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(phenylsulfonyl_reagent_pattern))
            for mol in reactants
        )
        
        # Look for protected pyrrole in products
        has_protected_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenylsulfonyl_pyrrole_pattern))
            for mol in products
        )
        
        return has_free_pyrrole_reactant and has_phenylsulfonyl_reagent and has_protected_product
    
    def _is_deprotection_reaction(self, reactants, products) -> bool:
        """Check if reaction removes phenylsulfonyl group from pyrrole"""
        # Look for protected pyrrole in reactants
        has_protected_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenylsulfonyl_pyrrole_pattern))
            for mol in reactants
        )
        
        # Look for free pyrrole in products
        has_free_pyrrole_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(self.free_pyrrole_pattern))
            for mol in products
        )
        
        # Check for harsh deprotection conditions (presence of strong base or acid)
        harsh_conditions = self._has_harsh_deprotection_conditions(reactants)
        
        return has_protected_reactant and has_free_pyrrole_product and harsh_conditions
    
    def _has_harsh_deprotection_conditions(self, reactants) -> bool:
        """Check for reagents indicating harsh deprotection conditions"""
        harsh_reagent_patterns = [
            "[OH-].[K+]",  # KOH
            "[OH-].[Na+]", # NaOH  
            "CC(C)(C)[O-].[K+]", # t-BuOK
            "[H-].[Na+]",  # NaH
            "CCN(CC)CC",   # TEA (triethylamine)
        ]
        
        for mol in reactants:
            for pattern in harsh_reagent_patterns:
                try:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        return True
                except:
                    continue
        return True  # Assume harsh conditions if we can't definitively identify mild ones
