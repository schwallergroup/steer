"""Generated evaluation code for: Terminal Boc protection after azide reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TerminalBocProtectionAfterAzideReduction(BaseScoring):
    """
    Evaluates routes where terminal Boc protection occurs after azide reduction.
    Checks for azide reduction followed by Boc protection of primary amine in final steps.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            # For terminal reactions, lower depth fractions are better
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Boc protection of primary amine
        that could follow azide reduction.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
                
            # Check if reaction involves Boc protection
            boc_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]([C])([C])[C]")  # Boc group
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")  # Primary amine
            
            # Check if product has Boc-protected amine
            has_boc_protected_amine = False
            for prod_mol in product_mols:
                if prod_mol.HasSubstructMatch(boc_pattern):
                    # Check if there's a Boc-NH pattern
                    boc_nh_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]([C])([C])[C]~[NH]")
                    if prod_mol.HasSubstructMatch(boc_nh_pattern):
                        has_boc_protected_amine = True
                        break
                        
            # Check if reactant has primary amine
            has_primary_amine_reactant = False
            for react_mol in reactant_mols:
                if react_mol.HasSubstructMatch(primary_amine_pattern):
                    has_primary_amine_reactant = True
                    break
                    
            if has_boc_protected_amine and has_primary_amine_reactant:
                # Additional check: verify this could be terminal by checking route depth
                return self._is_terminal_step(d)
                
            return False
            
        except Exception:
            return False
    
    def _is_terminal_step(self, d):
        """
        Check if this step could be terminal by examining if we're near leaves
        and if previous step could involve azide reduction.
        """
        # Check if this is one of the final steps (depth check)
        current_depth = d.get("depth", 0)
        if current_depth > 2:  # Not in final 2 steps
            return False
            
        # Look for potential azide reduction in recent history
        parent = d.get("parent")
        if parent:
            parent_rxn = parent.get("metadata", {}).get("mapped_reaction_smiles", "")
            if self._contains_azide_reduction(parent_rxn):
                return True
                
        # Also check if current reaction could be part of one-pot sequence
        current_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        return self._contains_azide_reduction(current_rxn)
    
    def _contains_azide_reduction(self, rxn_smiles):
        """Check if reaction involves azide reduction to amine."""
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
                
            azide_pattern = Chem.MolFromSmarts("[N]=[N+]=[N-]")  # Azide group
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")  # Primary amine
            
            has_azide_reactant = any(mol.HasSubstructMatch(azide_pattern) for mol in reactant_mols)
            has_amine_product = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in product_mols)
            
            return has_azide_reactant and has_amine_product
            
        except Exception:
            return False
