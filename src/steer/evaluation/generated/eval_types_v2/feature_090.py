"""Generated evaluation code for: Aryl chloride introduction then removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ArylChlorideFormBreak(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the introduction of aryl chloride followed by its removal.
    Detects formation of C-Cl bonds in aromatic context and subsequent breaking of those bonds.
    """
    
    def __init__(self, config):
        self.require_both = config.get("require_both", True)
        self.aromatic_cl_pattern = Chem.MolFromSmarts("[cH0:1][Cl:2]")  # Aromatic carbon with chlorine
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        cl_formation_found = False
        cl_breaking_found = False
        
        for rxn in reactions:
            if self.detect_cl_formation(rxn):
                cl_formation_found = True
            if self.detect_cl_breaking(rxn):
                cl_breaking_found = True
                
        if self.require_both:
            condition = cl_formation_found and cl_breaking_found
        else:
            condition = cl_formation_found or cl_breaking_found
            
        return condition, len(reactions)
    
    def detect_cl_formation(self, rxn):
        """Detect formation of aromatic C-Cl bond"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
            
        # Count aromatic chlorines in reactants vs products
        reactant_cl_count = sum(len(mol.GetSubstructMatches(self.aromatic_cl_pattern)) 
                               for mol in reactants)
        product_cl_count = sum(len(mol.GetSubstructMatches(self.aromatic_cl_pattern)) 
                              for mol in products)
        
        return product_cl_count > reactant_cl_count
    
    def detect_cl_breaking(self, rxn):
        """Detect breaking of aromatic C-Cl bond"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        if not all(reactants) or not all(products):
            return False
            
        # Count aromatic chlorines in reactants vs products
        reactant_cl_count = sum(len(mol.GetSubstructMatches(self.aromatic_cl_pattern)) 
                               for mol in reactants)
        product_cl_count = sum(len(mol.GetSubstructMatches(self.aromatic_cl_pattern)) 
                              for mol in products)
        
        return reactant_cl_count > product_cl_count
