"""Generated evaluation code for: Early protecting group deprotection before final substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupStrategy(BaseScoring):
    """
    Evaluates whether THP protecting group deprotection occurs before final substitution.
    Checks if THP ether removal happens one step before the final SN2 reaction.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.deprotection_timing = config["parameters"]["deprotection_timing"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # SMARTS patterns for THP ether and phenol
        self.thp_ether_pattern = "[OH1][CH1]1[CH2][CH2][CH2][CH2][OH0]1"
        self.phenol_pattern = "c[OH1]"
        self.thp_protected_phenol = "c[OH0][CH1]1[CH2][CH2][CH2][CH2][OH0]1"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not implemented
        else:
            # Earlier deprotection before final step is better
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves THP deprotection before final substitution"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if this is a deprotection reaction
        is_deprotection = self._is_thp_deprotection(reactants, products)
        
        if not is_deprotection:
            return False
            
        # Check if there's a subsequent substitution reaction
        return self._has_subsequent_substitution(d)
    
    def _is_thp_deprotection(self, reactants: str, products: str) -> bool:
        """Check if reaction involves THP ether deprotection to phenol"""
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Check for THP protected phenol in reactants
            has_protected = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.thp_protected_phenol)) 
                              for mol in reactant_mols if mol)
            
            # Check for free phenol in products
            has_deprotected = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)) 
                                for mol in product_mols if mol)
            
            # Check for THP byproduct (tetrahydropyran or related)
            thp_byproduct_pattern = "[CH1]1[CH2][CH2][CH2][CH2][OH0]1"
            has_thp_byproduct = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(thp_byproduct_pattern)) 
                                  for mol in product_mols if mol)
            
            return has_protected and has_deprotected and has_thp_byproduct
            
        except Exception:
            return False
    
    def _has_subsequent_substitution(self, current_node) -> bool:
        """Check if there's a substitution reaction in the immediate children"""
        children = getattr(current_node, 'children', [])
        
        for child in children:
            metadata = child.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if self._is_substitution_reaction(mapped_rxn):
                return True
                
        return False
    
    def _is_substitution_reaction(self, mapped_rxn: str) -> bool:
        """Check if reaction is a substitution (SN2) reaction"""
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            # Look for typical SN2 patterns: nucleophile + electrophile -> substituted product + leaving group
            leaving_groups = ["[Cl-]", "[Br-]", "[I-]", "OS(=O)(=O)C", "OS(=O)(=O)c"]
            
            has_leaving_group = any(
                any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(lg_pattern)) 
                    for mol in product_mols if mol)
                for lg_pattern in leaving_groups
            )
            
            # Check for carbon-heteroatom bond formation
            cn_bond_pattern = "[C][N]"
            co_bond_pattern = "[C][O]"
            cs_bond_pattern = "[C][S]"
            
            bond_formation_patterns = [cn_bond_pattern, co_bond_pattern, cs_bond_pattern]
            
            has_bond_formation = any(
                any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for mol in product_mols if mol)
                for pattern in bond_formation_patterns
            )
            
            return has_leaving_group and has_bond_formation
            
        except Exception:
            return False
