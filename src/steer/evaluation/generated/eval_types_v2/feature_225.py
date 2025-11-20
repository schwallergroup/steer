"""Generated evaluation code for: Benzyl ether protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates the use of benzyl ether protecting group strategy for phenols.
    Checks for:
    1. Presence of phenol protection with benzyl groups
    2. Subsequent reactions that benefit from protection
    3. Final deprotection via hydrogenolysis
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.require_deprotection = config.get("require_deprotection", True)
        self.require_coupling_reaction = config.get("require_coupling_reaction", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_benzyl_protection = any(self.detect_benzyl_protection(r) for r in reactions)
        has_deprotection = any(self.detect_benzyl_deprotection(r) for r in reactions)
        has_coupling = any(self.detect_coupling_reaction(r) for r in reactions)
        
        # Check if the strategy is properly implemented
        protection_ok = not self.require_protection or has_benzyl_protection
        deprotection_ok = not self.require_deprotection or has_deprotection
        coupling_ok = not self.require_coupling_reaction or has_coupling
        
        # Additional check: if protection occurs, deprotection should happen later
        sequence_ok = True
        if has_benzyl_protection and self.require_deprotection:
            protection_depth = self.find_reaction_depth(reactions, self.detect_benzyl_protection)
            deprotection_depth = self.find_reaction_depth(reactions, self.detect_benzyl_deprotection)
            sequence_ok = protection_depth > deprotection_depth  # Earlier in synthesis (higher depth)
        
        condition = protection_ok and deprotection_ok and coupling_ok and sequence_ok
        return condition, len(reactions)
    
    def detect_benzyl_protection(self, rxn):
        """Detect formation of benzyl ether from phenol"""
        prod_mol = Chem.MolFromSmiles(rxn.split(">>")[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn.split(">>")[1].split(".")]
        
        # Pattern for benzyl ether formation
        benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
        phenol_pattern = Chem.MolFromSmarts("c1ccccc1O")
        benzyl_halide_pattern = Chem.MolFromSmarts("c1ccccc1C[Cl,Br,I]")
        
        # Check if product has benzyl ether
        has_benzyl_ether = prod_mol and prod_mol.HasSubstructMatch(benzyl_ether_pattern)
        
        # Check if reactants have phenol and benzyl halide/alcohol
        has_phenol = any(mol and mol.HasSubstructMatch(phenol_pattern) for mol in react_mols)
        has_benzyl_electrophile = any(mol and (mol.HasSubstructMatch(benzyl_halide_pattern) or 
                                             mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1CO"))) 
                                    for mol in react_mols)
        
        return has_benzyl_ether and has_phenol and has_benzyl_electrophile
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect hydrogenolysis removal of benzyl ether to regenerate phenol"""
        prod_mol = Chem.MolFromSmiles(rxn.split(">>")[0])
        react_mols = [Chem.MolFromSmiles(r) for r in rxn.split(">>")[1].split(".")]
        
        benzyl_ether_pattern = Chem.MolFromSmarts("c1ccccc1COc2ccccc2")
        phenol_pattern = Chem.MolFromSmarts("c1ccccc1O")
        
        # Check if reactant has benzyl ether and product has phenol
        has_benzyl_ether_reactant = any(mol and mol.HasSubstructMatch(benzyl_ether_pattern) 
                                       for mol in react_mols)
        has_phenol_product = prod_mol and prod_mol.HasSubstructMatch(phenol_pattern)
        
        # Check for hydrogenolysis conditions (H2 present)
        has_hydrogen = any("H" in Chem.MolToSmiles(mol) and len(Chem.MolToSmiles(mol)) <= 3 
                          for mol in react_mols if mol)
        
        return has_benzyl_ether_reactant and has_phenol_product and has_hydrogen
    
    def detect_coupling_reaction(self, rxn):
        """Detect coupling reactions that would benefit from phenol protection"""
        # Look for common coupling reactions (Suzuki, Heck, etc.)
        suzuki_patterns = [
            Chem.MolFromSmarts("c1ccccc1B(O)O"),  # Boronic acid
            Chem.MolFromSmarts("c1ccccc1[B]"),     # Boron-containing
        ]
        
        halide_pattern = Chem.MolFromSmarts("c1ccccc1[Cl,Br,I]")  # Aryl halide
        
        react_mols = [Chem.MolFromSmiles(r) for r in rxn.split(">>")[1].split(".")]
        
        has_boron = any(mol and any(mol.HasSubstructMatch(pattern) for pattern in suzuki_patterns)
                       for mol in react_mols)
        has_halide = any(mol and mol.HasSubstructMatch(halide_pattern) for mol in react_mols)
        
        return has_boron and has_halide
    
    def find_reaction_depth(self, reactions, detection_func):
        """Find the depth at which a specific reaction type occurs"""
        for i, rxn in enumerate(reactions):
            if detection_func(rxn):
                return len(reactions) - i  # Convert to depth from end
        return -1
