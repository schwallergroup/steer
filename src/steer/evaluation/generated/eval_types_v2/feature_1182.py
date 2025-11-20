"""Generated evaluation code for: Multi-step trityl chloride preparation from chlorobenzene"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TritylChloridePreparation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for multi-step trityl chloride preparation from chlorobenzene.
    Checks for presence of Friedel-Crafts acylation, Grignard addition, and alcohol chlorination reactions.
    """
    
    def __init__(self, config):
        self.required_reactions = config["reaction_sequence"]
        self.target_reagent = config["target_reagent"]
        self.allow_friedel_crafts = "friedel_crafts_acylation" in self.required_reactions
        self.allow_grignard = "grignard_addition" in self.required_reactions
        self.allow_chlorination = "alcohol_chlorination" in self.required_reactions
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for each required reaction type
        has_friedel_crafts = any(self.detect_friedel_crafts_acylation(r) for r in reactions)
        has_grignard = any(self.detect_grignard_addition(r) for r in reactions)
        has_chlorination = any(self.detect_alcohol_chlorination(r) for r in reactions)
        
        # Check for trityl chloride formation
        has_trityl_chloride = any(self.detect_trityl_chloride_formation(r) for r in reactions)
        
        # All required reactions must be present
        condition = (has_friedel_crafts == self.allow_friedel_crafts and
                    has_grignard == self.allow_grignard and 
                    has_chlorination == self.allow_chlorination and
                    has_trityl_chloride)
        
        return condition, len(reactions)
    
    def detect_friedel_crafts_acylation(self, rxn):
        """Detect Friedel-Crafts acylation: aromatic + acyl chloride -> ketone"""
        # Look for formation of aromatic ketone from aromatic compound and acyl chloride
        ketone_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#8])-[#6]"
        acyl_chloride_pattern = "[#6]-[#6](=[#8])-[#17]"
        
        return self.detect_pattern_consumption_and_formation(rxn, acyl_chloride_pattern, ketone_pattern)
    
    def detect_grignard_addition(self, rxn):
        """Detect Grignard addition to ketone forming tertiary alcohol"""
        # Look for ketone + Grignard reagent -> tertiary alcohol
        grignard_pattern = "[#6]-[#12]"  # C-Mg bond
        ketone_pattern = "[#6](=[#8])"
        tertiary_alcohol_pattern = "[#6]([#8])([#6])([#6])[#6]"  # Tertiary alcohol
        
        rxn_parts = rxn.split(">>")
        reactants = rxn_parts[1] if len(rxn_parts) > 1 else ""
        products = rxn_parts[0]
        
        has_grignard_reactant = any(self.has_substructure(r, grignard_pattern) 
                                   for r in reactants.split(".") if r.strip())
        has_ketone_reactant = any(self.has_substructure(r, ketone_pattern)
                                 for r in reactants.split(".") if r.strip())
        has_tertiary_alcohol_product = any(self.has_substructure(p, tertiary_alcohol_pattern)
                                          for p in products.split(".") if p.strip())
        
        return has_grignard_reactant and has_ketone_reactant and has_tertiary_alcohol_product
    
    def detect_alcohol_chlorination(self, rxn):
        """Detect conversion of tertiary alcohol to tertiary chloride"""
        tertiary_alcohol_pattern = "[#6]([#8])([#6])([#6])[#6]"
        tertiary_chloride_pattern = "[#6]([#17])([#6])([#6])[#6]"
        
        return self.detect_pattern_consumption_and_formation(rxn, tertiary_alcohol_pattern, tertiary_chloride_pattern)
    
    def detect_trityl_chloride_formation(self, rxn):
        """Detect formation of trityl chloride specifically"""
        # Trityl chloride: three phenyl groups attached to carbon with chlorine
        trityl_chloride_pattern = "[#17]-[#6](-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1)(-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2)-[#6]3:[#6]:[#6]:[#6]:[#6]:[#6]:3"
        
        rxn_parts = rxn.split(">>")
        products = rxn_parts[0] if len(rxn_parts) > 0 else ""
        
        return any(self.has_substructure(p, trityl_chloride_pattern)
                  for p in products.split(".") if p.strip())
    
    def detect_pattern_consumption_and_formation(self, rxn, consumed_pattern, formed_pattern):
        """Helper method to detect consumption of one pattern and formation of another"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        has_consumed_reactant = any(self.has_substructure(r, consumed_pattern)
                                   for r in reactants.split(".") if r.strip())
        has_formed_product = any(self.has_substructure(p, formed_pattern)
                                for p in products.split(".") if p.strip())
        
        return has_consumed_reactant and has_formed_product
    
    def has_substructure(self, smiles, pattern):
        """Check if molecule contains substructure pattern"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return False
            return mol.HasSubstructMatch(pattern_mol)
        except:
            return False
