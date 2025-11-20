"""Generated evaluation code for: Two-step sulfur oxidation sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TwoStepSulfurOxidation(MultiRxnCondBase):
    """
    Evaluates whether a two-step sulfur oxidation sequence occurs in the synthesis route.
    Checks for sequential sulfide->sulfoxide->sulfone transformations rather than 
    direct sulfide->sulfone oxidation.
    """
    
    def __init__(self, config):
        self.require_sequential = config.get("sequential", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find all sulfur oxidation reactions
        sulfide_to_sulfoxide_rxns = []
        sulfoxide_to_sulfone_rxns = []
        direct_sulfone_rxns = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_sulfide_to_sulfoxide(rxn):
                sulfide_to_sulfoxide_rxns.append(i)
            elif self.detect_sulfoxide_to_sulfone(rxn):
                sulfoxide_to_sulfone_rxns.append(i)
            elif self.detect_direct_sulfone_formation(rxn):
                direct_sulfone_rxns.append(i)
        
        # Check if we have the two-step sequence
        has_two_step = len(sulfide_to_sulfoxide_rxns) > 0 and len(sulfoxide_to_sulfone_rxns) > 0
        
        # If sequential is required, check that we don't have direct sulfone formation
        # on the same sulfur atoms
        if self.require_sequential and has_two_step:
            condition = len(direct_sulfone_rxns) == 0
        else:
            condition = has_two_step
        
        return condition, len(reactions)
    
    def detect_sulfide_to_sulfoxide(self, rxn):
        """Detect sulfide to sulfoxide oxidation"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products = Chem.MolFromSmiles(rxn_parts[1])
            
            if reactants is None or products is None:
                return False
            
            # Pattern for sulfide: R-S-R
            sulfide_pattern = Chem.MolFromSmarts("[C,c]-[S;X2;v2]-[C,c]")
            # Pattern for sulfoxide: R-S(=O)-R  
            sulfoxide_pattern = Chem.MolFromSmarts("[C,c]-[S;X3;v4](=[O;X1])-[C,c]")
            
            has_sulfide_reactant = reactants.HasSubstructMatch(sulfide_pattern)
            has_sulfoxide_product = products.HasSubstructMatch(sulfoxide_pattern)
            
            return has_sulfide_reactant and has_sulfoxide_product
            
        except:
            return False
    
    def detect_sulfoxide_to_sulfone(self, rxn):
        """Detect sulfoxide to sulfone oxidation"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products = Chem.MolFromSmiles(rxn_parts[1])
            
            if reactants is None or products is None:
                return False
            
            # Pattern for sulfoxide: R-S(=O)-R
            sulfoxide_pattern = Chem.MolFromSmarts("[C,c]-[S;X3;v4](=[O;X1])-[C,c]")
            # Pattern for sulfone: R-S(=O)(=O)-R
            sulfone_pattern = Chem.MolFromSmarts("[C,c]-[S;X4;v6](=[O;X1])(=[O;X1])-[C,c]")
            
            has_sulfoxide_reactant = reactants.HasSubstructMatch(sulfoxide_pattern)
            has_sulfone_product = products.HasSubstructMatch(sulfone_pattern)
            
            return has_sulfoxide_reactant and has_sulfone_product
            
        except:
            return False
    
    def detect_direct_sulfone_formation(self, rxn):
        """Detect direct sulfide to sulfone oxidation (single step)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products = Chem.MolFromSmiles(rxn_parts[1])
            
            if reactants is None or products is None:
                return False
            
            # Pattern for sulfide: R-S-R
            sulfide_pattern = Chem.MolFromSmarts("[C,c]-[S;X2;v2]-[C,c]")
            # Pattern for sulfone: R-S(=O)(=O)-R
            sulfone_pattern = Chem.MolFromSmarts("[C,c]-[S;X4;v6](=[O;X1])(=[O;X1])-[C,c]")
            
            has_sulfide_reactant = reactants.HasSubstructMatch(sulfide_pattern)
            has_sulfone_product = products.HasSubstructMatch(sulfone_pattern)
            
            return has_sulfide_reactant and has_sulfone_product
            
        except:
            return False
