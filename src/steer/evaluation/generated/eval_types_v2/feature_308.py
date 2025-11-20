"""Generated evaluation code for: Integrated protecting group removal with oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IntegratedProtectingGroupRemoval(MultiRxnCondBase):
    """
    Evaluates synthesis routes for integrated protecting group removal with oxidation.
    Specifically checks for benzyl protection of sulfur that is removed during 
    oxidative conversion to sulfonate ester in one pot reaction.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "benzyl")
        self.protected_atom = config.get("protected_atom", "S")
        self.removal_integrated = config.get("removal_integrated", True)
        self.removal_reaction = config.get("removal_reaction", "oxidative_sulfonylation")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for integrated benzyl-sulfur protection removal with oxidation
        integrated_found = any(self.detect_integrated_removal(r) for r in reactions)
        
        condition = integrated_found == self.removal_integrated
        return condition, len(reactions)
    
    def detect_integrated_removal(self, rxn):
        """
        Detects if a reaction involves integrated benzyl protecting group removal
        from sulfur during oxidative sulfonylation.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not reactants or not products:
            return False
        
        # Check for benzyl-protected sulfur in reactants
        benzyl_sulfur_protected = False
        for mol in reactants:
            if mol and self.has_benzyl_protected_sulfur(mol):
                benzyl_sulfur_protected = True
                break
                
        # Check for sulfonate ester formation in products
        sulfonate_formed = False
        for mol in products:
            if mol and self.has_sulfonate_ester(mol):
                sulfonate_formed = True
                break
                
        # Check for benzyl group liberation (as benzyl alcohol or benzaldehyde)
        benzyl_liberated = False
        for mol in products:
            if mol and self.has_liberated_benzyl(mol):
                benzyl_liberated = True
                break
        
        return benzyl_sulfur_protected and sulfonate_formed and benzyl_liberated
    
    def has_benzyl_protected_sulfur(self, mol):
        """Check for benzyl group attached to sulfur."""
        # Benzyl-S pattern: phenyl-CH2-S
        pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH2]-[#16]")
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def has_sulfonate_ester(self, mol):
        """Check for sulfonate ester formation (R-SO2-OR')."""
        pattern = Chem.MolFromSmarts("[#16](=[O])(=[O])-[O]-[#6]")
        return mol.HasSubstructMatch(pattern) if pattern else False
    
    def has_liberated_benzyl(self, mol):
        """Check for liberated benzyl group (benzyl alcohol or benzaldehyde)."""
        benzyl_alcohol = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH2]-[OH]")
        benzaldehyde = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[CH]=O")
        
        has_alcohol = mol.HasSubstructMatch(benzyl_alcohol) if benzyl_alcohol else False
        has_aldehyde = mol.HasSubstructMatch(benzaldehyde) if benzaldehyde else False
        
        return has_alcohol or has_aldehyde
