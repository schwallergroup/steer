"""Generated evaluation code for: Early stage glycosylation with protecting groups"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GlycosylationProtectingGroups(BaseScoring):
    """
    Evaluates early stage glycosylation with protecting groups.
    Checks for acetate-protected glucose installation early in the route
    and deprotection in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.substrate = config["parameters"]["substrate"]
        self.timing = config["parameters"]["timing"]
        self.deprotection_timing = config["parameters"]["deprotection_timing"]
        
        # Define SMARTS patterns for acetate-protected glucose
        self.glucose_core = "[CH]1O[CH][CH]([OH,O[CH3],OC(=O)[CH3]])[CH]([OH,O[CH3],OC(=O)[CH3]])[CH]([OH,O[CH3],OC(=O)[CH3]])O1"
        self.acetate_pattern = "OC(=O)[CH3]"
        self.protected_glucose = "[CH]1O[CH][CH](OC(=O)[CH3])[CH](OC(=O)[CH3])[CH](OC(=O)[CH3])O1"
        self.trichloroacetimidate = "OC(=N)C(Cl)(Cl)Cl"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Glycosylation doesn't happen
        else:
            # Early stage glycosylation is better (lower depth)
            if x <= 0.3:  # Very early stage
                return 10
            elif x <= 0.5:  # Early stage
                return 8
            elif x <= 0.7:  # Mid stage
                return 5
            else:  # Late stage
                return 2
    
    def hit_condition(self, d):
        """Check if this reaction involves glycosylation with protecting groups"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn = rxn_smiles.split(">>")
            products = Chem.MolFromSmiles(rxn[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check if product contains protected glucose
            if not self._has_protected_glucose(products):
                return False
            
            # Check if glycosylation occurs (glucose donor in reactants)
            has_glucose_donor = False
            has_acceptor = False
            
            for reactant in reactants:
                if reactant is None:
                    continue
                    
                # Check for protected glucose donor (often with trichloroacetimidate)
                if (self._has_protected_glucose(reactant) and 
                    (self._has_leaving_group(reactant) or 
                     self._has_trichloroacetimidate(reactant))):
                    has_glucose_donor = True
                
                # Check for acceptor (has OH groups but not glucose)
                elif (self._has_alcohol_groups(reactant) and 
                      not self._has_glucose_core(reactant)):
                    has_acceptor = True
            
            return has_glucose_donor and has_acceptor
            
        except Exception:
            return False
    
    def _has_protected_glucose(self, mol):
        """Check if molecule contains acetate-protected glucose"""
        if mol is None:
            return False
        glucose_pattern = Chem.MolFromSmarts(self.protected_glucose)
        return mol.HasSubstructMatch(glucose_pattern)
    
    def _has_glucose_core(self, mol):
        """Check if molecule contains glucose core structure"""
        if mol is None:
            return False
        glucose_pattern = Chem.MolFromSmarts(self.glucose_core)
        return mol.HasSubstructMatch(glucose_pattern)
    
    def _has_trichloroacetimidate(self, mol):
        """Check for trichloroacetimidate leaving group"""
        if mol is None:
            return False
        pattern = Chem.MolFromSmarts(self.trichloroacetimidate)
        return mol.HasSubstructMatch(pattern)
    
    def _has_leaving_group(self, mol):
        """Check for common glycosyl donor leaving groups"""
        if mol is None:
            return False
        # Common leaving groups: halides, acetates at anomeric position
        leaving_groups = [
            "[CH]1O[CH]([Cl,Br,I,F])[CH][CH][CH]O1",  # Anomeric halide
            "[CH]1O[CH](OC(=O)[CH3])[CH][CH][CH]O1"   # Anomeric acetate
        ]
        
        for lg_smarts in leaving_groups:
            pattern = Chem.MolFromSmarts(lg_smarts)
            if mol.HasSubstructMatch(pattern):
                return True
        return False
    
    def _has_alcohol_groups(self, mol):
        """Check if molecule has alcohol groups (potential acceptor)"""
        if mol is None:
            return False
        alcohol_pattern = Chem.MolFromSmarts("[OH]")
        return mol.HasSubstructMatch(alcohol_pattern)
