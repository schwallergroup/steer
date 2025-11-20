"""Generated evaluation code for: Trifluoroacetyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrifluoroacetylProtectionStrategy(BaseScoring):
    """
    Evaluates synthesis routes based on the use of trifluoroacetyl (TFA) protecting group strategy.
    
    This class checks if a trifluoroacetyl protecting group is introduced at a specific depth
    in the synthesis route, typically used for protecting nitrogen atoms in piperidines.
    The TFA group requires harsh deprotection conditions, so earlier introduction may be penalized.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group_smarts = config["parameters"]["protecting_group_smarts"]
        self.atom_protected = config["parameters"]["atom_protected"]
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10 scale).
        Later introduction of harsh protecting groups is generally penalized.
        """
        if x < 0:
            return 0  # Protection strategy not used
        
        if self.condition_type == "bool":
            return 1  # Just checking if strategy is used
        else:
            # Penalize late-stage introduction of harsh protecting groups
            # Earlier protection (lower x) gets higher score
            return max(0, 1 - x)
    
    def hit_condition(self, d):
        """
        Check if trifluoroacetyl protecting group is introduced in this reaction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".") if p.strip()]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not products or not reactants:
                return False
            
            # Create pattern for TFA protecting group
            tfa_pattern = Chem.MolFromSmarts(self.protecting_group_smarts)
            if not tfa_pattern:
                return False
            
            # Check if TFA group appears in products but not in reactants
            tfa_in_products = any(mol.HasSubstructMatch(tfa_pattern) for mol in products if mol)
            tfa_in_reactants = any(mol.HasSubstructMatch(tfa_pattern) for mol in reactants if mol)
            
            # Protection reaction: TFA group introduced (present in products, absent in reactants)
            if tfa_in_products and not tfa_in_reactants:
                # Additional check: verify the protected atom is the specified type
                return self._verify_protection_site(reactants, products, tfa_pattern)
            
            return False
            
        except Exception:
            return False
    
    def _verify_protection_site(self, reactants, products, tfa_pattern):
        """
        Verify that the TFA group is protecting the specified atom type (e.g., nitrogen).
        """
        try:
            # Find unprotected atom in reactants that could be protected
            unprotected_pattern = Chem.MolFromSmarts(f"[{self.atom_protected}H]")  # e.g., [NH] for nitrogen
            if not unprotected_pattern:
                return True  # If we can't verify, assume it's correct
            
            # Check if we have unprotected sites in reactants
            has_unprotected_reactant = any(mol.HasSubstructMatch(unprotected_pattern) 
                                         for mol in reactants if mol)
            
            # Check if we have TFA-protected sites in products
            has_tfa_products = any(mol.HasSubstructMatch(tfa_pattern) 
                                 for mol in products if mol)
            
            return has_unprotected_reactant and has_tfa_products
            
        except Exception:
            return True  # If verification fails, assume the TFA detection is sufficient
