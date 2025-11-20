"""Generated evaluation code for: Benzyl group temporary installation and removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates benzyl protecting group strategy - checks for benzyl installation 
    followed by removal, particularly for amine protection.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.strategy_type = config.get("strategy_type", "temporary")
        self.context = config.get("context", "amine")
        
        # SMARTS patterns for benzyl-protected amines and related reactions
        self.benzyl_amine_pattern = "[CH2][c]1[cH][cH][cH][cH][cH]1"  # Bn-N
        self.benzyl_group_pattern = "[CH2][c]1[cH][cH][cH][cH][cH]1"
        self.free_amine_pattern = "[NH2,NH1,NH0]"
        
    def condition_depth(self, d):
        """Check if benzyl protecting group strategy is present in the route"""
        reactions = self.get_rxns(d)
        
        has_installation = any(self.detect_benzyl_installation(r) for r in reactions)
        has_removal = any(self.detect_benzyl_removal(r) for r in reactions)
        
        # Strategy is present if both installation and removal are found
        strategy_present = has_installation and has_removal
        
        return strategy_present, len(reactions)
    
    def detect_benzyl_installation(self, rxn):
        """Detect benzyl group installation (e.g., reductive amination)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactants have free amine and benzyl aldehyde/halide
            has_amine_reactant = any(self.has_free_amine(mol) for mol in reactants)
            has_benzyl_source = any(self.has_benzyl_electrophile(mol) for mol in reactants)
            
            # Check if product has benzyl-protected amine
            has_benzyl_amine_product = any(self.has_benzyl_amine(mol) for mol in products)
            
            return has_amine_reactant and has_benzyl_source and has_benzyl_amine_product
            
        except:
            return False
    
    def detect_benzyl_removal(self, rxn):
        """Detect benzyl group removal (debenzylation)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if reactant has benzyl-protected amine
            has_benzyl_amine_reactant = any(self.has_benzyl_amine(mol) for mol in reactants)
            
            # Check if product has free amine and potentially toluene/benzyl alcohol
            has_free_amine_product = any(self.has_free_amine(mol) for mol in products)
            has_benzyl_waste = any(self.is_benzyl_waste(mol) for mol in products)
            
            return has_benzyl_amine_reactant and has_free_amine_product
            
        except:
            return False
    
    def has_benzyl_amine(self, mol):
        """Check if molecule contains benzyl-protected amine"""
        if mol is None:
            return False
        benzyl_n_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0][CH2][c]1[cH][cH][cH][cH][cH]1")
        return mol.HasSubstructMatch(benzyl_n_pattern)
    
    def has_free_amine(self, mol):
        """Check if molecule contains free amine"""
        if mol is None:
            return False
        free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        return mol.HasSubstructMatch(free_amine_pattern)
    
    def has_benzyl_electrophile(self, mol):
        """Check if molecule is a benzyl electrophile (aldehyde, halide, etc.)"""
        if mol is None:
            return False
        # Benzaldehyde
        benzaldehyde = Chem.MolFromSmarts("[c]1[cH][cH][cH][cH][cH]1[CH]=O")
        # Benzyl halides
        benzyl_halide = Chem.MolFromSmarts("[c]1[cH][cH][cH][cH][cH]1[CH2][Cl,Br,I]")
        
        return mol.HasSubstructMatch(benzaldehyde) or mol.HasSubstructMatch(benzyl_halide)
    
    def is_benzyl_waste(self, mol):
        """Check if molecule is benzyl waste product (toluene, benzyl alcohol)"""
        if mol is None:
            return False
        toluene = Chem.MolFromSmarts("[c]1[cH][cH][cH][cH][cH]1[CH3]")
        benzyl_alcohol = Chem.MolFromSmarts("[c]1[cH][cH][cH][cH][cH]1[CH2][OH]")
        
        return mol.HasSubstructMatch(toluene) or mol.HasSubstructMatch(benzyl_alcohol)
