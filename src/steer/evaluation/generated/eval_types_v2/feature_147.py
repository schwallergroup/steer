"""Generated evaluation code for: Protecting group swap sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates routes for protecting group swap sequences, specifically carbamate-to-carbamate
    transformations where one protecting group is removed and another is installed on the same atom.
    """
    
    def __init__(self, config):
        self.swap_type = config.get("swap_type", "carbamate_to_carbamate")
        self.sequence_length = config.get("sequence_length", 3)
        self.protecting_groups = config.get("protecting_groups", ["methyl_carbamate", "boc"])
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "methyl_carbamate": "[NH1][C](=O)[O][CH3]",
            "boc": "[NH1][C](=O)[O][C]([CH3])([CH3])[CH3]",
            "cbz": "[NH1][C](=O)[O][CH2]c1ccccc1",
            "fmoc": "[NH1][C](=O)[O][CH2][CH]c1ccccc1c2ccccc2"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        swap_found = self.detect_protecting_group_swap(reactions)
        return swap_found, len(reactions)
    
    def detect_protecting_group_swap(self, reactions) -> bool:
        """
        Detects if there's a protecting group swap sequence in the reaction list.
        """
        if len(reactions) < 2:
            return False
            
        # Look for deprotection followed by protection within sequence_length steps
        for i in range(len(reactions) - 1):
            for j in range(i + 1, min(i + self.sequence_length, len(reactions))):
                if self.is_swap_sequence(reactions[i], reactions[j]):
                    return True
        return False
    
    def is_swap_sequence(self, rxn1, rxn2) -> bool:
        """
        Checks if two reactions form a protecting group swap sequence.
        """
        # Extract mapped nitrogen atoms from both reactions
        n_atoms_rxn1 = self.get_nitrogen_map_nums(rxn1)
        n_atoms_rxn2 = self.get_nitrogen_map_nums(rxn2)
        
        # Find common nitrogen atoms between reactions
        common_nitrogens = n_atoms_rxn1.intersection(n_atoms_rxn2)
        
        if not common_nitrogens:
            return False
            
        # Check if one reaction removes a protecting group and the other adds one
        for n_map in common_nitrogens:
            if self.is_deprotection_protection_pair(rxn1, rxn2, n_map):
                return True
        return False
    
    def get_nitrogen_map_nums(self, rxn) -> set:
        """
        Get atom map numbers for nitrogens involved in carbamate chemistry.
        """
        reactants_smi, products_smi = rxn.split(">>")
        nitrogen_maps = set()
        
        # Check products for carbamate nitrogens
        for prod_smi in products_smi.split("."):
            mol = Chem.MolFromSmiles(prod_smi)
            if mol:
                for pg_name, pattern in self.pg_patterns.items():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                        for match in matches:
                            # First atom in pattern is nitrogen
                            n_atom = mol.GetAtomWithIdx(match[0])
                            if n_atom.GetAtomMapNum() > 0:
                                nitrogen_maps.add(n_atom.GetAtomMapNum())
        
        # Also check reactants
        for react_smi in reactants_smi.split("."):
            mol = Chem.MolFromSmiles(react_smi)
            if mol:
                for pg_name, pattern in self.pg_patterns.items():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                        for match in matches:
                            n_atom = mol.GetAtomWithIdx(match[0])
                            if n_atom.GetAtomMapNum() > 0:
                                nitrogen_maps.add(n_atom.GetAtomMapNum())
        
        return nitrogen_maps
    
    def is_deprotection_protection_pair(self, rxn1, rxn2, nitrogen_map) -> bool:
        """
        Check if the two reactions form a deprotection-protection pair on the specified nitrogen.
        """
        rxn1_removes_pg = self.removes_protecting_group(rxn1, nitrogen_map)
        rxn1_adds_pg = self.adds_protecting_group(rxn1, nitrogen_map)
        rxn2_removes_pg = self.removes_protecting_group(rxn2, nitrogen_map)
        rxn2_adds_pg = self.adds_protecting_group(rxn2, nitrogen_map)
        
        # Check for deprotection-protection sequence
        return (rxn1_removes_pg and rxn2_adds_pg) or (rxn2_removes_pg and rxn1_adds_pg)
    
    def removes_protecting_group(self, rxn, nitrogen_map) -> bool:
        """
        Check if reaction removes a protecting group from the specified nitrogen.
        """
        reactants_smi, products_smi = rxn.split(">>")
        
        # Check if nitrogen has protecting group in reactants but not products
        has_pg_reactants = self.nitrogen_has_protecting_group(reactants_smi, nitrogen_map)
        has_pg_products = self.nitrogen_has_protecting_group(products_smi, nitrogen_map)
        
        return has_pg_reactants and not has_pg_products
    
    def adds_protecting_group(self, rxn, nitrogen_map) -> bool:
        """
        Check if reaction adds a protecting group to the specified nitrogen.
        """
        reactants_smi, products_smi = rxn.split(">>")
        
        # Check if nitrogen lacks protecting group in reactants but has it in products
        has_pg_reactants = self.nitrogen_has_protecting_group(reactants_smi, nitrogen_map)
        has_pg_products = self.nitrogen_has_protecting_group(products_smi, nitrogen_map)
        
        return not has_pg_reactants and has_pg_products
    
    def nitrogen_has_protecting_group(self, smiles, nitrogen_map) -> bool:
        """
        Check if a nitrogen with given map number has any of the target protecting groups.
        """
        for mol_smi in smiles.split("."):
            mol = Chem.MolFromSmiles(mol_smi)
            if mol:
                for pg_name in self.protecting_groups:
                    pattern = self.pg_patterns.get(pg_name)
                    if pattern and mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                        for match in matches:
                            n_atom = mol.GetAtomWithIdx(match[0])
                            if n_atom.GetAtomMapNum() == nitrogen_map:
                                return True
        return False
