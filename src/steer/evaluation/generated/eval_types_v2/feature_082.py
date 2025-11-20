"""Generated evaluation code for: Trifluoroacetyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrifluoroacetylProtection(BaseScoring):
    """
    Evaluates the use of trifluoroacetyl protecting group strategy for secondary amines.
    Checks if trifluoroacetyl protection is applied to secondary amines for selective protection,
    particularly useful in controlling regioselectivity in piperidine acylation reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Protection strategy not used
        
        if self.condition_type == "bool":
            return 10  # Strategy is present
        else:
            # Earlier protection is generally better (lower depth)
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if trifluoroacetyl protection of secondary amine occurs in this reaction"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for trifluoroacetyl protection reaction
            return self._is_trifluoroacetyl_protection(reactants, products)
            
        except Exception:
            return False
    
    def _is_trifluoroacetyl_protection(self, reactants, products) -> bool:
        """Check if reaction involves trifluoroacetyl protection of secondary amine"""
        # Trifluoroacetyl patterns
        tfa_reagent_pattern = Chem.MolFromSmarts("[C](=[O])[C]([F])([F])([F])")  # CF3CO-
        tfa_chloride_pattern = Chem.MolFromSmarts("[C](=[O])([Cl])[C]([F])([F])([F])")  # CF3COCl
        tfa_anhydride_pattern = Chem.MolFromSmarts("[C](=[O])([O][C](=[O])[C]([F])([F])([F]))[C]([F])([F])([F])")  # (CF3CO)2O
        
        # Secondary amine pattern (not in amide)
        sec_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]([#6])[#6]")
        
        # Protected secondary amine (trifluoroacetamide)
        protected_amine_pattern = Chem.MolFromSmarts("[NX3;H0]([#6])([#6])[C](=[O])[C]([F])([F])([F])")
        
        # Check if reactants contain trifluoroacetyl reagent and secondary amine
        has_tfa_reagent = False
        has_sec_amine = False
        
        for mol in reactants:
            if (tfa_chloride_pattern and mol.HasSubstructMatch(tfa_chloride_pattern)) or \
               (tfa_anhydride_pattern and mol.HasSubstructMatch(tfa_anhydride_pattern)) or \
               (tfa_reagent_pattern and mol.HasSubstructMatch(tfa_reagent_pattern)):
                has_tfa_reagent = True
            
            if sec_amine_pattern and mol.HasSubstructMatch(sec_amine_pattern):
                has_sec_amine = True
        
        # Check if products contain protected amine
        has_protected_amine = False
        for mol in products:
            if protected_amine_pattern and mol.HasSubstructMatch(protected_amine_pattern):
                has_protected_amine = True
                break
        
        # Additional check for piperidine context (mentioned in rationale)
        has_piperidine_context = False
        piperidine_pattern = Chem.MolFromSmarts("[N]1[CH2][CH2][CH2][CH2][CH2]1")
        
        for mol in reactants + products:
            if piperidine_pattern and mol.HasSubstructMatch(piperidine_pattern):
                has_piperidine_context = True
                break
        
        # Return true if we have protection reaction, with bonus for piperidine context
        return (has_tfa_reagent and has_sec_amine and has_protected_amine) or \
               (has_protected_amine and has_piperidine_context)
