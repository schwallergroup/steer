"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectingGroup(MultiRxnCondBase):
    """
    Evaluates the use of benzyl ether protecting group strategy for phenols.
    Checks for protection of phenol with benzyl group and subsequent deprotection.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl_ether")
        self.functional_group = config.get("functional_group", "phenol")
        self.required_count = config.get("count", 1)
        
        # SMARTS patterns
        self.phenol_pattern = "[OH1][c]"  # Phenol OH
        self.benzyl_ether_pattern = "[c][CH2][O][c]"  # Benzyl ether linkage
        self.benzyl_protection_pattern = "[c][CH2][Br,Cl]"  # Benzyl halide for protection
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        protection_count = 0
        deprotection_count = 0
        
        for rxn in reactions:
            if self.detect_benzyl_protection(rxn):
                protection_count += 1
            elif self.detect_benzyl_deprotection(rxn):
                deprotection_count += 1
        
        # Strategy requires at least one protection-deprotection pair
        complete_cycles = min(protection_count, deprotection_count)
        condition = complete_cycles >= self.required_count
        
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """
        Detects benzyl protection of phenol: ArOH + BnX -> ArOBn + HX
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactants contain phenol and benzyl halide
        has_phenol = False
        has_benzyl_halide = False
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)):
                        has_phenol = True
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_protection_pattern)):
                        has_benzyl_halide = True
            except:
                continue
        
        # Check if products contain benzyl ether
        has_benzyl_ether = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)):
                    has_benzyl_ether = True
                    break
            except:
                continue
        
        return has_phenol and has_benzyl_halide and has_benzyl_ether
    
    def detect_benzyl_deprotection(self, rxn):
        """
        Detects benzyl deprotection: ArOBn -> ArOH + Bn fragments
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if reactants contain benzyl ether
        has_benzyl_ether = False
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)):
                    has_benzyl_ether = True
                    break
            except:
                continue
        
        # Check if products contain regenerated phenol
        has_phenol = False
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.phenol_pattern)):
                    has_phenol = True
                    break
            except:
                continue
        
        return has_benzyl_ether and has_phenol
