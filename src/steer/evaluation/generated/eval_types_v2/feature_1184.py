"""Generated evaluation code for: Multi-step trityl chloride reagent synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TritylChlorideReagentSynthesis(MultiRxnCondBase):
    """
    Evaluates routes that synthesize trityl chloride reagent in multiple steps
    before using it in a final coupling/protection reaction.
    """
    
    def __init__(self, config):
        self.reagent_synthesis_steps = config.get("reagent_synthesis_steps", 3)
        self.final_coupling_step = config.get("final_coupling_step", 1)
        self.total_steps = config.get("total_steps", 4)
        
        # Trityl chloride SMARTS pattern
        self.trityl_chloride_pattern = "ClC(c1ccccc1)(c2ccccc2)c3ccccc3"
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Check if we have the expected number of steps
        if total_reactions != self.total_steps:
            return False, total_reactions
            
        # Find trityl chloride synthesis reactions
        trityl_synthesis_reactions = []
        final_coupling_reaction = None
        
        for i, rxn in enumerate(reactions):
            if self.involves_trityl_chloride_synthesis(rxn):
                trityl_synthesis_reactions.append(i)
            elif self.involves_trityl_coupling(rxn):
                final_coupling_reaction = i
                
        # Check if we have the right number of trityl synthesis steps
        if len(trityl_synthesis_reactions) != self.reagent_synthesis_steps:
            return False, total_reactions
            
        # Check if we have exactly one final coupling step
        if final_coupling_reaction is None:
            return False, total_reactions
            
        # Verify that trityl synthesis happens before final coupling
        # (lower indices in BFS order indicate earlier steps)
        if any(idx >= final_coupling_reaction for idx in trityl_synthesis_reactions):
            return False, total_reactions
            
        return True, total_reactions
        
    def involves_trityl_chloride_synthesis(self, rxn):
        """Check if reaction involves synthesis of trityl chloride moiety"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check if trityl chloride appears in products but not fully formed in reactants
        trityl_in_products = any(self.contains_trityl_chloride(p) for p in products)
        
        if not trityl_in_products:
            return False
            
        # Check for C-C coupling to form trityl structure or chlorination
        return self.is_trityl_formation_reaction(reactants, products)
        
    def involves_trityl_coupling(self, rxn):
        """Check if reaction uses pre-formed trityl chloride as a reagent"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Trityl chloride should be in reactants
        trityl_in_reactants = any(self.contains_trityl_chloride(r) for r in reactants)
        
        if not trityl_in_reactants:
            return False
            
        # Products should contain trityl ether/ester (protection product)
        return any(self.contains_trityl_protected_product(p) for p in products)
        
    def contains_trityl_chloride(self, smiles):
        """Check if molecule contains trityl chloride moiety"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            pattern = Chem.MolFromSmarts(self.trityl_chloride_pattern)
            return mol.HasSubstructMatch(pattern) if pattern else False
        except:
            return False
            
    def contains_trityl_protected_product(self, smiles):
        """Check if molecule contains trityl ether or ester (protection product)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
                
            # Trityl ether pattern: R-O-C(Ph)3
            trityl_ether_pattern = Chem.MolFromSmarts("OC(c1ccccc1)(c2ccccc2)c3ccccc3")
            # Trityl ester pattern: R-CO-O-C(Ph)3  
            trityl_ester_pattern = Chem.MolFromSmarts("C(=O)OC(c1ccccc1)(c2ccccc2)c3ccccc3")
            
            return (mol.HasSubstructMatch(trityl_ether_pattern) if trityl_ether_pattern else False) or \
                   (mol.HasSubstructMatch(trityl_ester_pattern) if trityl_ester_pattern else False)
        except:
            return False
            
    def is_trityl_formation_reaction(self, reactants, products):
        """Check if reaction forms trityl structure through C-C coupling or chlorination"""
        try:
            # Look for benzene rings in reactants vs trityl structure in products
            benzene_pattern = Chem.MolFromSmarts("c1ccccc1")
            chloride_pattern = Chem.MolFromSmarts("[Cl-]")
            
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants if Chem.MolFromSmiles(r)]
            
            # Count benzene rings in reactants
            total_benzene_rings = sum(len(mol.GetSubstructMatches(benzene_pattern)) 
                                    for mol in reactant_mols if benzene_pattern)
            
            # Check for chloride source
            has_chloride = any(mol.HasSubstructMatch(chloride_pattern) 
                             for mol in reactant_mols if chloride_pattern)
            
            # Trityl formation typically requires 3+ benzene rings and chloride source
            return total_benzene_rings >= 3 or has_chloride
        except:
            return False
