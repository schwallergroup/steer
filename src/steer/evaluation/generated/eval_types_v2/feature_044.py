"""Generated evaluation code for: Sequential protecting group swap acetate to benzyl"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates whether a route performs sequential protecting group swap from acetate to benzyl
    on the same primary alcohol, with the reactions occurring consecutively.
    """
    
    def __init__(self, config):
        self.sequence = config["sequence"]  # ["acetate_deprotection", "benzyl_protection"]
        self.functional_group = config["functional_group"]  # "primary_alcohol"
        self.consecutive = config["consecutive"]  # True
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find acetate deprotection and benzyl protection reactions
        acetate_deprotection_indices = []
        benzyl_protection_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_acetate_deprotection(rxn):
                acetate_deprotection_indices.append(i)
            if self.detect_benzyl_protection(rxn):
                benzyl_protection_indices.append(i)
        
        # Check if we have both reaction types
        if not acetate_deprotection_indices or not benzyl_protection_indices:
            return False, len(reactions)
        
        # Check for consecutive sequence
        if self.consecutive:
            for deprotect_idx in acetate_deprotection_indices:
                for protect_idx in benzyl_protection_indices:
                    # Check if benzyl protection immediately follows acetate deprotection
                    if protect_idx == deprotect_idx + 1:
                        # Verify they act on the same primary alcohol position
                        if self.same_alcohol_position(reactions[deprotect_idx], reactions[protect_idx]):
                            return True, len(reactions)
        else:
            # Just check if both reactions exist and act on same position
            for deprotect_idx in acetate_deprotection_indices:
                for protect_idx in benzyl_protection_indices:
                    if self.same_alcohol_position(reactions[deprotect_idx], reactions[protect_idx]):
                        return True, len(reactions)
        
        return False, len(reactions)
    
    def detect_acetate_deprotection(self, rxn):
        """Detect acetate deprotection reaction"""
        rxn_parts = rxn.split(">>")
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Pattern for acetate group
        acetate_pattern = Chem.MolFromSmarts("[CH3]C(=O)O[CH2]")
        # Pattern for primary alcohol
        alcohol_pattern = Chem.MolFromSmarts("[CH2]O")
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactants)
            product_mol = Chem.MolFromSmiles(products.split(".")[0])  # Main product
            
            if reactant_mol is None or product_mol is None:
                return False
            
            # Check if reactant has acetate and product has free alcohol
            has_acetate_reactant = reactant_mol.HasSubstructMatch(acetate_pattern)
            has_alcohol_product = product_mol.HasSubstructMatch(alcohol_pattern)
            
            return has_acetate_reactant and has_alcohol_product
            
        except:
            return False
    
    def detect_benzyl_protection(self, rxn):
        """Detect benzyl ether protection reaction"""
        rxn_parts = rxn.split(">>")
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Pattern for primary alcohol
        alcohol_pattern = Chem.MolFromSmarts("[CH2]O")
        # Pattern for benzyl ether
        benzyl_ether_pattern = Chem.MolFromSmarts("[CH2]Oc1ccccc1")
        
        try:
            reactant_mol = Chem.MolFromSmiles(reactants.split(".")[0])  # Main reactant
            product_mol = Chem.MolFromSmiles(products)
            
            if reactant_mol is None or product_mol is None:
                return False
            
            # Check if reactant has free alcohol and product has benzyl ether
            has_alcohol_reactant = reactant_mol.HasSubstructMatch(alcohol_pattern)
            has_benzyl_product = product_mol.HasSubstructMatch(benzyl_ether_pattern)
            
            return has_alcohol_reactant and has_benzyl_product
            
        except:
            return False
    
    def same_alcohol_position(self, rxn1, rxn2):
        """
        Check if two reactions act on the same primary alcohol position
        by comparing atom mapping numbers
        """
        try:
            # Get mapped reaction molecules
            rxn1_parts = rxn1.split(">>")
            rxn2_parts = rxn2.split(">>")
            
            # For rxn1 (acetate deprotection), check product for alcohol mapping
            rxn1_product = Chem.MolFromSmiles(rxn2_parts[1].split(".")[0])
            
            # For rxn2 (benzyl protection), check reactant for alcohol mapping  
            rxn2_reactant = Chem.MolFromSmiles(rxn2_parts[0].split(".")[0])
            
            if rxn1_product is None or rxn2_reactant is None:
                return False
            
            # Get atom mapping numbers for oxygen atoms in primary alcohols
            def get_alcohol_oxygen_maps(mol):
                maps = []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'O' and atom.GetAtomMapNum() > 0:
                        # Check if connected to a carbon that could be primary alcohol
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetSymbol() == 'C':
                                maps.append(atom.GetAtomMapNum())
                                break
                return maps
            
            rxn1_maps = get_alcohol_oxygen_maps(rxn1_product)
            rxn2_maps = get_alcohol_oxygen_maps(rxn2_reactant)
            
            # Check for overlap in mapping numbers
            return bool(set(rxn1_maps) & set(rxn2_maps))
            
        except:
            return True  # Default to True if mapping analysis fails
