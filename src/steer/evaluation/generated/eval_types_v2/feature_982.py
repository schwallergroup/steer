"""Generated evaluation code for: Dual benzyl protecting groups with selectivity issues"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualBenzylProtectingGroups(MultiRxnCondBase):
    """
    Detects routes using both N-benzyl and O-benzyl protecting groups,
    which create selectivity issues during hydrogenation removal.
    """
    
    def __init__(self, config):
        self.selectivity_required = config.get("selectivity_required", True)
        self.removal_method = config.get("removal_method", "hydrogenation")
        
        # SMARTS patterns for benzyl protecting groups
        self.n_benzyl_pattern = "[#7]-[CH2]-c1ccccc1"  # N-benzyl
        self.o_benzyl_pattern = "[#8]-[CH2]-c1ccccc1"  # O-benzyl
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_n_benzyl = False
        has_o_benzyl = False
        
        # Check all reactions for benzyl protecting group formation or presence
        for rxn in reactions:
            if self.detect_n_benzyl_formation(rxn) or self.detect_n_benzyl_presence(rxn):
                has_n_benzyl = True
            if self.detect_o_benzyl_formation(rxn) or self.detect_o_benzyl_presence(rxn):
                has_o_benzyl = True
                
        # Problem exists if both protecting groups are present
        dual_benzyl_problem = has_n_benzyl and has_o_benzyl
        
        if self.selectivity_required:
            # Return True if selectivity problem exists (dual benzyl groups)
            condition = dual_benzyl_problem
        else:
            # Return True if either protecting group is used
            condition = has_n_benzyl or has_o_benzyl
            
        return condition, len(reactions)
    
    def detect_n_benzyl_formation(self, rxn):
        """Detect formation of N-benzyl protecting groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if N-benzyl is formed (absent in reactants, present in products)
        reactant_has_n_benzyl = self.has_substructure_in_smiles(reactants, self.n_benzyl_pattern)
        product_has_n_benzyl = self.has_substructure_in_smiles(products, self.n_benzyl_pattern)
        
        return not reactant_has_n_benzyl and product_has_n_benzyl
    
    def detect_o_benzyl_formation(self, rxn):
        """Detect formation of O-benzyl protecting groups"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if O-benzyl is formed (absent in reactants, present in products)
        reactant_has_o_benzyl = self.has_substructure_in_smiles(reactants, self.o_benzyl_pattern)
        product_has_o_benzyl = self.has_substructure_in_smiles(products, self.o_benzyl_pattern)
        
        return not reactant_has_o_benzyl and product_has_o_benzyl
    
    def detect_n_benzyl_presence(self, rxn):
        """Detect presence of N-benzyl groups in reaction products"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[1]
        return self.has_substructure_in_smiles(products, self.n_benzyl_pattern)
    
    def detect_o_benzyl_presence(self, rxn):
        """Detect presence of O-benzyl groups in reaction products"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[1]
        return self.has_substructure_in_smiles(products, self.o_benzyl_pattern)
    
    def has_substructure_in_smiles(self, smiles_str, pattern):
        """Check if any molecule in SMILES string contains the substructure"""
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
                
            # Handle multiple molecules separated by dots
            molecules = smiles_str.split(".")
            for mol_smiles in molecules:
                mol = Chem.MolFromSmiles(mol_smiles.strip())
                if mol is not None and mol.HasSubstructMatch(pattern_mol):
                    return True
            return False
        except:
            return False
