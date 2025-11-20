"""Generated evaluation code for: Dual benzyl ether protecting group usage"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualBenzylEtherProtection(MultiRxnCondBase):
    """
    Evaluates routes based on the simultaneous use of two benzyl ether protecting groups
    on both a secondary alcohol and a phenol substrate.
    """
    
    def __init__(self, config):
        self.count = config.get("count", 2)
        self.substrate_types = config.get("substrate_types", ["secondary_alcohol", "phenol"])
        self.allow_dual_benzyl = config.get("allow_dual_benzyl", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Track benzyl ether formations on different substrate types
        secondary_alcohol_benzyl = 0
        phenol_benzyl = 0
        
        for rxn in reactions:
            if self.detect_benzyl_ether_formation(rxn):
                substrate_type = self.identify_substrate_type(rxn)
                if substrate_type == "secondary_alcohol":
                    secondary_alcohol_benzyl += 1
                elif substrate_type == "phenol":
                    phenol_benzyl += 1
        
        # Check if we have dual benzyl ether protection as specified
        has_dual_protection = (
            secondary_alcohol_benzyl >= 1 and 
            phenol_benzyl >= 1 and
            (secondary_alcohol_benzyl + phenol_benzyl) >= self.count
        )
        
        condition = has_dual_protection == self.allow_dual_benzyl
        return condition, len(reactions)
    
    def detect_benzyl_ether_formation(self, rxn):
        """Detect formation of benzyl ether (C-O-CH2-Ph) bonds"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        product = Chem.MolFromSmiles(rxn_parts[1].strip())
        
        if not all([product] + reactants):
            return False
        
        # Benzyl ether pattern: aromatic ring connected to CH2-O-
        benzyl_ether_pattern = Chem.MolFromSmarts("[cH1,c:1][c:2]1[cH1,c][cH1,c][cH1,c][cH1,c][cH1,c]1[CH2:3][O:4]")
        
        if not product.HasSubstructMatch(benzyl_ether_pattern):
            return False
        
        # Check if this pattern is newly formed (not present in reactants)
        for reactant in reactants:
            if reactant.HasSubstructMatch(benzyl_ether_pattern):
                return False
                
        return True
    
    def identify_substrate_type(self, rxn):
        """Identify if the substrate being protected is a secondary alcohol or phenol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return None
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        
        # Secondary alcohol pattern: [C][CH]([C])[OH]
        secondary_alcohol_pattern = Chem.MolFromSmarts("[C][CH]([C])[OH]")
        
        # Phenol pattern: aromatic OH
        phenol_pattern = Chem.MolFromSmarts("[OH][c]")
        
        for reactant in reactants:
            if not reactant:
                continue
                
            # Check for phenol first (more specific)
            if reactant.HasSubstructMatch(phenol_pattern):
                return "phenol"
            elif reactant.HasSubstructMatch(secondary_alcohol_pattern):
                return "secondary_alcohol"
                
        return None
