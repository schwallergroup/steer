"""Generated evaluation code for: DMTr protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DMTrProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that use DMTr (dimethoxytrityl) protection followed by deprotection
    on the same functional group position, specifically targeting primary alcohols.
    """
    
    def __init__(self, config):
        self.protecting_group = config["protecting_group"]
        self.protection_count = config["protection_count"]
        self.deprotection_count = config["deprotection_count"]
        self.target_functional_group = config["target_functional_group"]
        
        # DMTr substructure pattern
        self.dmtr_pattern = Chem.MolFromSmarts("[CH2:1]-O-C(c1ccc(OC)cc1)(c2ccc(OC)cc2)(c3ccccc3)")
        # Primary alcohol pattern  
        self.primary_alcohol_pattern = Chem.MolFromSmarts("[CH2:1]-OH")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track DMTr protection and deprotection events with atom mapping
        protection_events = []
        deprotection_events = []
        
        for rxn in reactions:
            protection_atoms = self.detect_dmtr_protection(rxn)
            deprotection_atoms = self.detect_dmtr_deprotection(rxn)
            
            if protection_atoms:
                protection_events.extend(protection_atoms)
            if deprotection_atoms:
                deprotection_events.extend(deprotection_atoms)
        
        # Check if we have the required number of protection/deprotection cycles
        protection_condition = len(protection_events) >= self.protection_count
        deprotection_condition = len(deprotection_events) >= self.deprotection_count
        
        # Check if protection and deprotection occur on the same position
        cycling_condition = False
        if protection_events and deprotection_events:
            # Look for matching atom map numbers between protection and deprotection
            protected_atoms = set(protection_events)
            deprotected_atoms = set(deprotection_events)
            cycling_condition = bool(protected_atoms.intersection(deprotected_atoms))
        
        overall_condition = protection_condition and deprotection_condition and cycling_condition
        
        return overall_condition, len(reactions)
    
    def detect_dmtr_protection(self, rxn):
        """Detect DMTr protection reactions - primary alcohol to DMTr ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
        
        reactants = [Chem.MolFromSmiles(smi) for smi in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(smi) for smi in rxn_parts[1].split(".")]
        
        protected_atoms = []
        
        # Look for primary alcohol in reactants and DMTr ether in products
        for reactant in reactants:
            if reactant and reactant.HasSubstructMatch(self.primary_alcohol_pattern):
                matches = reactant.GetSubstructMatches(self.primary_alcohol_pattern)
                for match in matches:
                    alcohol_atom_map = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                    
                    # Check if this atom becomes DMTr protected in products
                    for product in products:
                        if product and product.HasSubstructMatch(self.dmtr_pattern):
                            dmtr_matches = product.GetSubstructMatches(self.dmtr_pattern)
                            for dmtr_match in dmtr_matches:
                                dmtr_atom_map = product.GetAtomWithIdx(dmtr_match[0]).GetAtomMapNum()
                                if alcohol_atom_map == dmtr_atom_map and alcohol_atom_map != 0:
                                    protected_atoms.append(alcohol_atom_map)
        
        return protected_atoms
    
    def detect_dmtr_deprotection(self, rxn):
        """Detect DMTr deprotection reactions - DMTr ether to primary alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return []
        
        reactants = [Chem.MolFromSmiles(smi) for smi in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(smi) for smi in rxn_parts[1].split(".")]
        
        deprotected_atoms = []
        
        # Look for DMTr ether in reactants and primary alcohol in products
        for reactant in reactants:
            if reactant and reactant.HasSubstructMatch(self.dmtr_pattern):
                matches = reactant.GetSubstructMatches(self.dmtr_pattern)
                for match in matches:
                    dmtr_atom_map = reactant.GetAtomWithIdx(match[0]).GetAtomMapNum()
                    
                    # Check if this atom becomes a primary alcohol in products
                    for product in products:
                        if product and product.HasSubstructMatch(self.primary_alcohol_pattern):
                            alcohol_matches = product.GetSubstructMatches(self.primary_alcohol_pattern)
                            for alcohol_match in alcohol_matches:
                                alcohol_atom_map = product.GetAtomWithIdx(alcohol_match[0]).GetAtomMapNum()
                                if dmtr_atom_map == alcohol_atom_map and dmtr_atom_map != 0:
                                    deprotected_atoms.append(dmtr_atom_map)
        
        return deprotected_atoms
