"""Generated evaluation code for: Protecting group cycling around lactam nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ProtectingGroupCycling(MultiRxnCondBase):
    """
    Detects protecting group cycling around lactam nitrogen.
    Checks if a Boc protecting group is added and removed from the same lactam nitrogen
    within a specified number of steps.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.functional_group = config.get("functional_group", "lactam_nitrogen")
        self.max_steps_between = config.get("steps_between", 2) + 1  # Include endpoints
        
        # Define SMARTS patterns
        self.lactam_patterns = [
            "[NH1]C(=O)",  # General lactam nitrogen
            "[NH1]1CCCC(=O)1",  # 5-membered lactam (pyrrolidinone)
            "[NH1]1CCCCC(=O)1",  # 6-membered lactam (piperidinone)
            "[NH1]1CC(=O)C1",  # 4-membered lactam (azetidinone)
            "[NH1]1CCC(=O)CC1"  # 6-membered lactam alt pattern
        ]
        
        self.boc_patterns = [
            "N([CH3]?)C(=O)OC(C)(C)C",  # Boc-protected nitrogen
            "NC(=O)OC(C)(C)C"  # Simple Boc pattern
        ]
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track atom mappings and protecting group states
        cycling_detected = self.detect_protection_cycling(reactions)
        
        return cycling_detected, len(reactions)
    
    def detect_protection_cycling(self, reactions):
        """Detect if the same lactam nitrogen gets protected and deprotected"""
        protection_events = []
        
        for i, rxn in enumerate(reactions):
            mapped_rxn = rxn.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn:
                continue
                
            parts = mapped_rxn.split(">>")
            if len(parts) != 2:
                continue
                
            reactants = parts[0]
            products = parts[1]
            
            # Check for protection event (lactam -> Boc-lactam)
            if self.is_protection_event(reactants, products):
                lactam_atoms = self.get_lactam_nitrogen_maps(reactants)
                protection_events.append(("protect", i, lactam_atoms))
            
            # Check for deprotection event (Boc-lactam -> lactam)
            elif self.is_deprotection_event(reactants, products):
                lactam_atoms = self.get_lactam_nitrogen_maps(products)
                protection_events.append(("deprotect", i, lactam_atoms))
        
        # Look for cycling patterns
        return self.find_cycling_pattern(protection_events)
    
    def is_protection_event(self, reactants, products):
        """Check if lactam nitrogen gets Boc-protected"""
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
        
        if not all(reactant_mols + product_mols):
            return False
        
        # Check if reactants have lactam and products have Boc-protected lactam
        has_lactam_reactant = any(self.has_lactam_nitrogen(mol) for mol in reactant_mols if mol)
        has_boc_product = any(self.has_boc_protection(mol) for mol in product_mols if mol)
        
        return has_lactam_reactant and has_boc_product
    
    def is_deprotection_event(self, reactants, products):
        """Check if Boc-protected lactam nitrogen gets deprotected"""
        reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
        
        if not all(reactant_mols + product_mols):
            return False
        
        # Check if reactants have Boc-protected and products have free lactam
        has_boc_reactant = any(self.has_boc_protection(mol) for mol in reactant_mols if mol)
        has_lactam_product = any(self.has_lactam_nitrogen(mol) for mol in product_mols if mol)
        
        return has_boc_reactant and has_lactam_product
    
    def has_lactam_nitrogen(self, mol):
        """Check if molecule contains lactam nitrogen"""
        if not mol:
            return False
        for pattern in self.lactam_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def has_boc_protection(self, mol):
        """Check if molecule contains Boc-protected nitrogen"""
        if not mol:
            return False
        for pattern in self.boc_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        return False
    
    def get_lactam_nitrogen_maps(self, smiles):
        """Get atom map numbers of lactam nitrogens"""
        mols = [Chem.MolFromSmiles(smi.strip()) for smi in smiles.split(".")]
        lactam_maps = []
        
        for mol in mols:
            if not mol:
                continue
            for pattern in self.lactam_patterns:
                matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
                for match in matches:
                    n_atom = mol.GetAtomWithIdx(match[0])  # First atom in pattern is N
                    if n_atom.GetAtomMapNum():
                        lactam_maps.append(n_atom.GetAtomMapNum())
        
        return lactam_maps
    
    def find_cycling_pattern(self, events):
        """Find if there's a protect-deprotect cycle within max_steps_between"""
        for i, (event1_type, step1, atoms1) in enumerate(events):
            for j, (event2_type, step2, atoms2) in enumerate(events[i+1:], i+1):
                # Check if it's a protect-deprotect cycle
                if event1_type == "protect" and event2_type == "deprotect":
                    # Check if steps are within allowed range
                    if abs(step2 - step1) <= self.max_steps_between:
                        # Check if same nitrogen atoms are involved
                        if set(atoms1) & set(atoms2):  # Intersection of atom sets
                            return True
        return False
