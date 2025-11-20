"""Generated evaluation code for: Protecting group cycling TFA to Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects protecting group cycling where a TFA group is removed and a Boc group
    is added to the same functional group (nitrogen) in the same route.
    """
    
    def __init__(self, config):
        self.deprotection_group = config["parameters"]["deprotection_group"]
        self.protection_group = config["parameters"]["protection_group"]
        self.same_functional_group = config["parameters"]["same_functional_group"]
        
        # SMARTS patterns for detecting protecting groups
        self.tfa_pattern = Chem.MolFromSmarts("[NX3][C](=O)C(F)(F)F")  # TFA on nitrogen
        self.boc_pattern = Chem.MolFromSmarts("[NX3][C](=O)OC(C)(C)C")  # Boc on nitrogen
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        tfa_removal_atoms = set()
        boc_addition_atoms = set()
        
        # Check each reaction for TFA removal or Boc addition
        for rxn in reactions:
            tfa_atoms = self.detect_tfa_removal(rxn)
            boc_atoms = self.detect_boc_addition(rxn)
            
            tfa_removal_atoms.update(tfa_atoms)
            boc_addition_atoms.update(boc_atoms)
        
        # Check if there's overlap in atom map numbers (same nitrogen)
        if self.same_functional_group:
            cycling_detected = bool(tfa_removal_atoms.intersection(boc_addition_atoms))
        else:
            cycling_detected = bool(tfa_removal_atoms) and bool(boc_addition_atoms)
        
        return cycling_detected, len(reactions)
    
    def detect_tfa_removal(self, rxn):
        """Detect TFA group removal and return atom map numbers of affected nitrogens"""
        affected_atoms = set()
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return affected_atoms
            
        reactant = Chem.MolFromSmiles(rxn_parts[0])
        products = [Chem.MolFromSmiles(p) for p in rxn_parts[1].split(".")]
        
        if not reactant or not all(products):
            return affected_atoms
        
        # Find TFA groups in reactant
        if reactant.HasSubstructMatch(self.tfa_pattern):
            matches = reactant.GetSubstructMatches(self.tfa_pattern)
            
            for match in matches:
                nitrogen_idx = match[0]  # First atom in pattern is nitrogen
                nitrogen_atom = reactant.GetAtomWithIdx(nitrogen_idx)
                atom_map_num = nitrogen_atom.GetAtomMapNum()
                
                # Check if this nitrogen loses TFA in products
                tfa_lost = True
                for product in products:
                    if self._has_tfa_on_mapped_atom(product, atom_map_num):
                        tfa_lost = False
                        break
                
                if tfa_lost and atom_map_num > 0:
                    affected_atoms.add(atom_map_num)
        
        return affected_atoms
    
    def detect_boc_addition(self, rxn):
        """Detect Boc group addition and return atom map numbers of affected nitrogens"""
        affected_atoms = set()
        
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return affected_atoms
            
        reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[0].split(".")]
        product = Chem.MolFromSmiles(rxn_parts[1])
        
        if not all(reactants) or not product:
            return affected_atoms
        
        # Find Boc groups in product
        if product.HasSubstructMatch(self.boc_pattern):
            matches = product.GetSubstructMatches(self.boc_pattern)
            
            for match in matches:
                nitrogen_idx = match[0]  # First atom in pattern is nitrogen
                nitrogen_atom = product.GetAtomWithIdx(nitrogen_idx)
                atom_map_num = nitrogen_atom.GetAtomMapNum()
                
                # Check if this nitrogen gains Boc (wasn't present in reactants)
                boc_added = True
                for reactant in reactants:
                    if self._has_boc_on_mapped_atom(reactant, atom_map_num):
                        boc_added = False
                        break
                
                if boc_added and atom_map_num > 0:
                    affected_atoms.add(atom_map_num)
        
        return affected_atoms
    
    def _has_tfa_on_mapped_atom(self, mol, atom_map_num):
        """Check if molecule has TFA on atom with given map number"""
        if not mol or atom_map_num <= 0:
            return False
            
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == atom_map_num and atom.GetSymbol() == 'N':
                atom_idx = atom.GetIdx()
                matches = mol.GetSubstructMatches(self.tfa_pattern)
                for match in matches:
                    if match[0] == atom_idx:
                        return True
        return False
    
    def _has_boc_on_mapped_atom(self, mol, atom_map_num):
        """Check if molecule has Boc on atom with given map number"""
        if not mol or atom_map_num <= 0:
            return False
            
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == atom_map_num and atom.GetSymbol() == 'N':
                atom_idx = atom.GetIdx()
                matches = mol.GetSubstructMatches(self.boc_pattern)
                for match in matches:
                    if match[0] == atom_idx:
                        return True
        return False
