"""Generated evaluation code for: Protecting group cycling acetate to MOM ether"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes for inefficient protecting group cycling from acetate to MOM ether.
    Detects sequences where an acetate-protected alcohol is deprotected and then
    immediately reprotected as a MOM ether, indicating poor synthetic planning.
    """
    
    def __init__(self, config):
        self.protection_sequence = config.get("protection_sequence", ["acetate", "free_alcohol", "MOM_ether"])
        self.functional_group = config.get("functional_group", "primary_alcohol")
        self.cycling = config.get("cycling", True)
        
        # SMARTS patterns for detection
        self.acetate_pattern = "[OH1][C](=O)[CH3]"  # Acetate ester
        self.mom_ether_pattern = "[OH1][CH2][OH1][CH3]"  # MOM ether
        self.free_alcohol_pattern = "[CH2][OH1]"  # Primary alcohol
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route contains the acetate->free alcohol->MOM cycling pattern"""
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Look for the cycling pattern in consecutive reactions
        cycling_detected = False
        
        for i in range(len(reactions) - 1):
            current_rxn = reactions[i]
            next_rxn = reactions[i + 1]
            
            # Check if first reaction deprotects acetate to free alcohol
            if self.is_acetate_deprotection(current_rxn):
                # Check if next reaction protects as MOM ether
                if self.is_mom_protection(next_rxn):
                    # Verify same alcohol is involved by checking atom mapping
                    if self.same_alcohol_involved(current_rxn, next_rxn):
                        cycling_detected = True
                        break
        
        return cycling_detected, len(reactions)
    
    def is_acetate_deprotection(self, rxn):
        """Check if reaction removes acetate protection to give free alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = Chem.MolFromSmiles(rxn_parts[0])
        products_smiles = rxn_parts[1].split(".")
        
        if not reactants:
            return False
            
        # Check if reactant has acetate group
        has_acetate = reactants.HasSubstructMatch(Chem.MolFromSmarts(self.acetate_pattern))
        
        if not has_acetate:
            return False
            
        # Check if any product has free alcohol
        has_free_alcohol = False
        for prod_smi in products_smiles:
            prod = Chem.MolFromSmiles(prod_smi)
            if prod and prod.HasSubstructMatch(Chem.MolFromSmarts(self.free_alcohol_pattern)):
                has_free_alcohol = True
                break
                
        return has_free_alcohol
    
    def is_mom_protection(self, rxn):
        """Check if reaction protects free alcohol as MOM ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0].split(".")
        products = Chem.MolFromSmiles(rxn_parts[1])
        
        if not products:
            return False
            
        # Check if any reactant has free alcohol
        has_free_alcohol = False
        for react_smi in reactants_smiles:
            react = Chem.MolFromSmiles(react_smi)
            if react and react.HasSubstructMatch(Chem.MolFromSmarts(self.free_alcohol_pattern)):
                has_free_alcohol = True
                break
                
        if not has_free_alcohol:
            return False
            
        # Check if product has MOM ether
        has_mom = products.HasSubstructMatch(Chem.MolFromSmarts(self.mom_ether_pattern))
        
        return has_mom
    
    def same_alcohol_involved(self, rxn1, rxn2):
        """Check if the same alcohol carbon is involved in both reactions using atom mapping"""
        try:
            # Parse both reactions
            rxn1_parts = rxn1.split(">>")
            rxn2_parts = rxn2.split(">>")
            
            # Get the alcohol carbon atom map from first reaction product
            rxn1_products = rxn1_parts[1].split(".")
            alcohol_map_num = None
            
            for prod_smi in rxn1_products:
                mol = Chem.MolFromSmiles(prod_smi)
                if mol:
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() == 'C' and atom.GetAtomMapNum() > 0:
                            # Check if this carbon is connected to OH
                            for neighbor in atom.GetNeighbors():
                                if neighbor.GetSymbol() == 'O':
                                    alcohol_map_num = atom.GetAtomMapNum()
                                    break
                            if alcohol_map_num:
                                break
                if alcohol_map_num:
                    break
            
            if not alcohol_map_num:
                return True  # Assume same if can't determine (conservative)
                
            # Check if same atom map appears in second reaction reactants
            rxn2_reactants = rxn2_parts[0].split(".")
            for react_smi in rxn2_reactants:
                mol = Chem.MolFromSmiles(react_smi)
                if mol:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum() == alcohol_map_num:
                            return True
                            
            return False
            
        except:
            return True  # Conservative assumption if parsing fails
