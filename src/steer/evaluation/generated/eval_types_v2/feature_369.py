"""Generated evaluation code for: Orthogonal protecting group strategy TBDMS and Boc"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates synthesis routes for orthogonal protecting group strategies.
    Checks if specified protecting groups (TBDMS and Boc) are used with
    their corresponding orthogonal deprotection conditions.
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["TBDMS", "Boc"])
        self.orthogonal = config.get("orthogonal", True)
        self.deprotection_conditions = config.get("deprotection_conditions", ["fluoride", "acid"])
        
        # SMARTS patterns for protecting groups
        self.tbdms_pattern = "[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]"  # TBDMS pattern
        self.boc_pattern = "[#6](=[O])[O][#6]([CH3])([CH3])[CH3]"      # Boc pattern
        
        # Keywords for deprotection conditions
        self.fluoride_keywords = ["tbaf", "fluoride", "hf", "csf"]
        self.acid_keywords = ["tfa", "hcl", "acid", "trifluoroacetic"]

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track protection and deprotection events
        tbdms_protection = False
        tbdms_deprotection = False
        boc_protection = False
        boc_deprotection = False
        
        # Check each reaction for protection/deprotection patterns
        for rxn in reactions:
            if self.detect_tbdms_protection(rxn):
                tbdms_protection = True
            elif self.detect_tbdms_deprotection(rxn):
                tbdms_deprotection = True
                
            if self.detect_boc_protection(rxn):
                boc_protection = True
            elif self.detect_boc_deprotection(rxn):
                boc_deprotection = True
        
        # Evaluate orthogonal strategy
        if self.orthogonal:
            # Both protecting groups should be used and properly deprotected
            condition = (tbdms_protection and tbdms_deprotection and 
                        boc_protection and boc_deprotection)
        else:
            # At least one protecting group strategy should be complete
            condition = ((tbdms_protection and tbdms_deprotection) or 
                        (boc_protection and boc_deprotection))
        
        return condition, len(reactions)

    def detect_tbdms_protection(self, rxn):
        """Detect TBDMS protection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if TBDMS appears in products but not in reactants
        return (self.contains_tbdms(products) and 
                not self.contains_tbdms(reactants))

    def detect_tbdms_deprotection(self, rxn):
        """Detect TBDMS deprotection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if TBDMS appears in reactants but not in products
        # and fluoride conditions are present
        return (self.contains_tbdms(reactants) and 
                not self.contains_tbdms(products) and
                self.contains_fluoride_conditions(rxn))

    def detect_boc_protection(self, rxn):
        """Detect Boc protection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if Boc appears in products but not in reactants
        return (self.contains_boc(products) and 
                not self.contains_boc(reactants))

    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check if Boc appears in reactants but not in products
        # and acidic conditions are present
        return (self.contains_boc(reactants) and 
                not self.contains_boc(products) and
                self.contains_acid_conditions(rxn))

    def contains_tbdms(self, smiles_string):
        """Check if SMILES contains TBDMS group"""
        try:
            for smiles in smiles_string.split('.'):
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.tbdms_pattern)):
                    return True
        except:
            pass
        return False

    def contains_boc(self, smiles_string):
        """Check if SMILES contains Boc group"""
        try:
            for smiles in smiles_string.split('.'):
                mol = Chem.MolFromSmiles(smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.boc_pattern)):
                    return True
        except:
            pass
        return False

    def contains_fluoride_conditions(self, rxn):
        """Check if reaction contains fluoride-based deprotection conditions"""
        rxn_lower = rxn.lower()
        return any(keyword in rxn_lower for keyword in self.fluoride_keywords)

    def contains_acid_conditions(self, rxn):
        """Check if reaction contains acidic deprotection conditions"""
        rxn_lower = rxn.lower()
        return any(keyword in rxn_lower for keyword in self.acid_keywords)
