"""Generated evaluation code for: Dual protecting group strategy on amine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates the presence of dual protecting group strategy on amines.
    Checks if both Boc and trichloroethyl protecting groups are used
    on amine functionalities within the synthesis route.
    """
    
    def __init__(self, config):
        self.functional_group = config.get("functional_group", "amine")
        self.protecting_groups = config.get("protecting_groups", ["Boc", "trichloroethyl"])
        self.strategy = config.get("strategy", "dual_protection")
        
        # Define SMARTS patterns for protecting groups
        self.boc_pattern = Chem.MolFromSmarts("[N;!H0,!H1,!H2]-C(=O)-O-C(C)(C)C")  # Boc protection
        self.trichloroethyl_pattern = Chem.MolFromSmarts("[N;!H0,!H1,!H2]-C(=O)-O-C-C(Cl)(Cl)Cl")  # Trichloroethyl protection
        
        # Pattern for free amine that could be protected
        self.free_amine_pattern = Chem.MolFromSmarts("[N;H1,H2;!$(N-C=O)]")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_boc_protection = False
        has_trichloroethyl_protection = False
        has_dual_strategy = False
        
        for rxn in reactions:
            # Check if this reaction involves Boc protection
            if self.detect_boc_protection(rxn):
                has_boc_protection = True
            
            # Check if this reaction involves trichloroethyl protection
            if self.detect_trichloroethyl_protection(rxn):
                has_trichloroethyl_protection = True
            
            # Check if both protecting groups are present in same molecule
            if self.detect_dual_protection(rxn):
                has_dual_strategy = True
                break
        
        # Strategy is successful if both protecting groups are used
        condition = has_boc_protection and has_trichloroethyl_protection and has_dual_strategy
        
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection reaction on amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactants have free amine and products have Boc-protected amine
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r)]
        product_mols = [Chem.MolFromSmiles(p) for p in products if Chem.MolFromSmiles(p)]
        
        has_free_amine_reactant = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactant_mols)
        has_boc_product = any(mol.HasSubstructMatch(self.boc_pattern) for mol in product_mols)
        
        return has_free_amine_reactant and has_boc_product
    
    def detect_trichloroethyl_protection(self, rxn):
        """Detect trichloroethyl protection reaction on amine"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r)]
        product_mols = [Chem.MolFromSmiles(p) for p in products if Chem.MolFromSmiles(p)]
        
        has_free_amine_reactant = any(mol.HasSubstructMatch(self.free_amine_pattern) for mol in reactant_mols)
        has_trichloroethyl_product = any(mol.HasSubstructMatch(self.trichloroethyl_pattern) for mol in product_mols)
        
        return has_free_amine_reactant and has_trichloroethyl_product
    
    def detect_dual_protection(self, rxn):
        """Detect if both protecting groups are present on same molecule"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        all_mols = []
        for part in rxn_parts:
            for smiles in part.split("."):
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    all_mols.append(mol)
        
        # Check if any molecule has both protecting groups
        for mol in all_mols:
            has_boc = mol.HasSubstructMatch(self.boc_pattern)
            has_trichloroethyl = mol.HasSubstructMatch(self.trichloroethyl_pattern)
            if has_boc and has_trichloroethyl:
                return True
        
        return False
