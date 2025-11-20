"""Generated evaluation code for: Protecting group cycling with acetate-benzyl swap"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects protecting group cycling where acetate deprotection is followed by 
    benzyl protection and then benzyl deprotection, resulting in no net transformation.
    """
    
    def __init__(self, config):
        self.protection_sequence = config.get("protection_sequence", [
            "acetate_deprotection", 
            "benzyl_protection", 
            "benzyl_deprotection"
        ])
        self.min_steps = config.get("steps_involved", 3)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find sequence of protecting group operations
        pg_operations = []
        for rxn in reactions:
            if self.detect_acetate_deprotection(rxn):
                pg_operations.append("acetate_deprotection")
            elif self.detect_benzyl_protection(rxn):
                pg_operations.append("benzyl_protection")
            elif self.detect_benzyl_deprotection(rxn):
                pg_operations.append("benzyl_deprotection")
        
        # Check if the sequence matches the expected cycling pattern
        condition = self.has_cycling_pattern(pg_operations)
        return condition, len(reactions)
    
    def has_cycling_pattern(self, operations):
        """Check if operations contain the specified protecting group cycling sequence"""
        if len(operations) < self.min_steps:
            return False
            
        # Look for the sequence in order (not necessarily consecutive)
        seq_idx = 0
        for op in operations:
            if seq_idx < len(self.protection_sequence) and op == self.protection_sequence[seq_idx]:
                seq_idx += 1
                
        return seq_idx == len(self.protection_sequence)
    
    def detect_acetate_deprotection(self, rxn):
        """Detect acetate deprotection reactions"""
        reactants, products = self.split_reaction(rxn)
        
        # Acetate group pattern
        acetate_pattern = Chem.MolFromSmarts("[#6]C(=O)O[#6]")
        acetic_acid_pattern = Chem.MolFromSmarts("CC(=O)O")
        
        # Check if acetate is present in reactants but not products
        # and acetic acid or acetate ion is formed
        has_acetate_reactant = any(mol.HasSubstructMatch(acetate_pattern) for mol in reactants)
        has_acetic_acid_product = any(mol.HasSubstructMatch(acetic_acid_pattern) for mol in products)
        
        return has_acetate_reactant and has_acetic_acid_product
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl protection reactions"""
        reactants, products = self.split_reaction(rxn)
        
        # Free hydroxyl/amine pattern and benzyl ether/amine pattern
        benzyl_reagent = Chem.MolFromSmarts("c1ccccc1C[Cl,Br,I]")  # Benzyl halide
        benzyl_ether = Chem.MolFromSmarts("c1ccccc1CO[#6]")  # Benzyl ether
        benzyl_amine = Chem.MolFromSmarts("c1ccccc1CN[#6]")  # Benzyl amine
        
        has_benzyl_reagent = any(mol.HasSubstructMatch(benzyl_reagent) for mol in reactants)
        has_benzyl_product = any(mol.HasSubstructMatch(benzyl_ether) or mol.HasSubstructMatch(benzyl_amine) 
                                for mol in products)
        
        return has_benzyl_reagent and has_benzyl_product
    
    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl deprotection reactions"""
        reactants, products = self.split_reaction(rxn)
        
        # Benzyl ether/amine patterns
        benzyl_ether = Chem.MolFromSmarts("c1ccccc1CO[#6]")
        benzyl_amine = Chem.MolFromSmarts("c1ccccc1CN[#6]")
        toluene = Chem.MolFromSmarts("c1ccccc1C")  # Toluene byproduct
        
        has_benzyl_reactant = any(mol.HasSubstructMatch(benzyl_ether) or mol.HasSubstructMatch(benzyl_amine)
                                 for mol in reactants)
        has_toluene_product = any(mol.HasSubstructMatch(toluene) for mol in products)
        
        return has_benzyl_reactant and has_toluene_product
    
    def split_reaction(self, rxn):
        """Split mapped reaction SMILES into reactants and products"""
        rxn_parts = rxn.split(">>")
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
        products = [Chem.MolFromSmiles(rxn_parts[0].strip())]
        
        # Filter out None molecules
        reactants = [mol for mol in reactants if mol is not None]
        products = [mol for mol in products if mol is not None]
        
        return reactants, products
