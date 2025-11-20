"""Generated evaluation code for: TFA protecting group for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TFAProtectingGroupStrategy(BaseScoring):
    """
    Evaluates synthesis routes for the use of trifluoroacetyl (TFA) protecting group
    on piperidine nitrogen. Rewards routes that employ TFA protection for selective
    amide coupling without interference from the piperidine nitrogen.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "relative")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # TFA protection not used
        else:
            # Earlier use of TFA protection is better for synthetic planning
            if self.condition_type == "bool":
                return 1  # TFA protection found
            else:
                return 1 - x  # Earlier protection gets higher score
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves TFA protection or deprotection of piperidine nitrogen"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants, products = rxn_smiles.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        # Remove None molecules (invalid SMILES)
        reactant_mols = [mol for mol in reactant_mols if mol is not None]
        product_mols = [mol for mol in product_mols if mol is not None]
        
        # Check for TFA protection (reactant has free piperidine N, product has TFA-protected N)
        if self._has_tfa_protection(reactant_mols, product_mols):
            return True
            
        # Check for TFA deprotection (reactant has TFA-protected N, product has free N)
        if self._has_tfa_deprotection(reactant_mols, product_mols):
            return True
            
        return False
    
    def _has_tfa_protection(self, reactants, products) -> bool:
        """Check if reaction involves TFA protection of piperidine nitrogen"""
        # Piperidine pattern (free secondary amine in ring)
        piperidine_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][NH1][#6][#6]1")
        # TFA-protected piperidine pattern
        tfa_protected_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]N([C](=O)C(F)(F)F)[#6][#6]1")
        # TFA reagent pattern
        tfa_reagent_pattern = Chem.MolFromSmarts("FC(F)(F)C(=O)*")
        
        # Check if reactants contain free piperidine and TFA reagent
        has_free_piperidine = any(mol.HasSubstructMatch(piperidine_pattern) for mol in reactants)
        has_tfa_reagent = any(mol.HasSubstructMatch(tfa_reagent_pattern) for mol in reactants)
        
        # Check if products contain TFA-protected piperidine
        has_tfa_protected = any(mol.HasSubstructMatch(tfa_protected_pattern) for mol in products)
        
        return has_free_piperidine and (has_tfa_reagent or has_tfa_protected)
    
    def _has_tfa_deprotection(self, reactants, products) -> bool:
        """Check if reaction involves TFA deprotection of piperidine nitrogen"""
        # TFA-protected piperidine pattern
        tfa_protected_pattern = Chem.MolFromSmarts("[#6]1[#6][#6]N([C](=O)C(F)(F)F)[#6][#6]1")
        # Free piperidine pattern
        piperidine_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][NH1][#6][#6]1")
        
        # Check if reactants contain TFA-protected piperidine
        has_tfa_protected = any(mol.HasSubstructMatch(tfa_protected_pattern) for mol in reactants)
        
        # Check if products contain free piperidine
        has_free_piperidine = any(mol.HasSubstructMatch(piperidine_pattern) for mol in products)
        
        return has_tfa_protected and has_free_piperidine
