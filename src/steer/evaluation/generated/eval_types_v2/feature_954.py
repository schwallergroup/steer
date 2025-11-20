"""Generated evaluation code for: Sequential protecting group swap strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialProtectingGroupSwap(MultiRxnCondBase):
    """
    Evaluates routes for sequential protecting group swap strategy.
    Checks if Boc deprotection is immediately followed by SEM protection
    on the same imidazole nitrogen position.
    """
    
    def __init__(self, config):
        self.groups = config.get("groups", ["Boc", "SEM"])
        self.location = config.get("location", "imidazole_nitrogen")
        self.strategy_type = config.get("strategy_type", "sequential_swap")
        
        # Define SMARTS patterns for protecting groups
        self.boc_pattern = Chem.MolFromSmarts("[#7]C(=O)OC(C)(C)C")  # Boc group
        self.sem_pattern = Chem.MolFromSmarts("[#7]COC[Si](C)(C)C")  # SEM group
        self.imidazole_pattern = Chem.MolFromSmarts("c1cnc[nH]1")  # Imidazole
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if sequential protecting group swap occurs in the route"""
        reactions = self.get_rxns(d)
        
        # Find Boc deprotection and SEM protection reactions
        boc_deprotection_steps = []
        sem_protection_steps = []
        
        for i, rxn in enumerate(reactions):
            if self.is_boc_deprotection(rxn):
                atom_map = self.get_affected_nitrogen_map(rxn, self.boc_pattern)
                if atom_map and self.has_imidazole_nitrogen(rxn, atom_map):
                    boc_deprotection_steps.append((i, atom_map))
                    
            if self.is_sem_protection(rxn):
                atom_map = self.get_affected_nitrogen_map(rxn, self.sem_pattern)
                if atom_map and self.has_imidazole_nitrogen(rxn, atom_map):
                    sem_protection_steps.append((i, atom_map))
        
        # Check for sequential swap (Boc removal followed by SEM addition)
        sequential_swap_found = False
        for boc_step, boc_atom in boc_deprotection_steps:
            for sem_step, sem_atom in sem_protection_steps:
                # Check if SEM protection follows Boc deprotection on same atom
                if (sem_step == boc_step + 1 and boc_atom == sem_atom):
                    sequential_swap_found = True
                    break
            if sequential_swap_found:
                break
                
        return sequential_swap_found, len(reactions)
    
    def is_boc_deprotection(self, rxn):
        """Check if reaction involves Boc group removal"""
        rxn_parts = rxn.split(">>")
        reactant = Chem.MolFromSmiles(rxn_parts[0])
        product = Chem.MolFromSmiles(rxn_parts[1].split(".")[0])  # Main product
        
        if not reactant or not product:
            return False
            
        # Boc present in reactant but not in product
        return (reactant.HasSubstructMatch(self.boc_pattern) and 
                not product.HasSubstructMatch(self.boc_pattern))
    
    def is_sem_protection(self, rxn):
        """Check if reaction involves SEM group addition"""
        rxn_parts = rxn.split(">>")
        reactant = Chem.MolFromSmiles(rxn_parts[0])
        product = Chem.MolFromSmiles(rxn_parts[1].split(".")[0])  # Main product
        
        if not reactant or not product:
            return False
            
        # SEM not present in reactant but present in product
        return (not reactant.HasSubstructMatch(self.sem_pattern) and 
                product.HasSubstructMatch(self.sem_pattern))
    
    def get_affected_nitrogen_map(self, rxn, pattern):
        """Get atom map number of nitrogen affected by protecting group change"""
        rxn_parts = rxn.split(">>")
        reactant = Chem.MolFromSmiles(rxn_parts[0])
        
        if not reactant:
            return None
            
        matches = reactant.GetSubstructMatches(pattern)
        if matches:
            # Get the nitrogen atom from the first match
            nitrogen_idx = matches[0][0]  # First atom in pattern is nitrogen
            nitrogen_atom = reactant.GetAtomWithIdx(nitrogen_idx)
            return nitrogen_atom.GetAtomMapNum()
        return None
    
    def has_imidazole_nitrogen(self, rxn, target_map_num):
        """Check if the mapped nitrogen is part of an imidazole ring"""
        rxn_parts = rxn.split(">>")
        reactant = Chem.MolFromSmiles(rxn_parts[0])
        
        if not reactant:
            return False
            
        # Find atom with target map number
        target_atom = None
        for atom in reactant.GetAtoms():
            if atom.GetAtomMapNum() == target_map_num:
                target_atom = atom
                break
                
        if not target_atom or target_atom.GetSymbol() != 'N':
            return False
            
        # Check if this nitrogen is part of an imidazole
        imidazole_matches = reactant.GetSubstructMatches(self.imidazole_pattern)
        target_idx = target_atom.GetIdx()
        
        for match in imidazole_matches:
            if target_idx in match:
                return True
                
        return False
