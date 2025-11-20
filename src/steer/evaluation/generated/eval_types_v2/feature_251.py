"""Generated evaluation code for: Benzylic activation via bromination then hydrolysis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylicActivation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for benzylic activation via bromination then hydrolysis.
    Looks for sequential conversion of benzylic C-H to C-Br to C-OH.
    """
    
    def __init__(self, config):
        self.require_sequential = config.get("require_sequential", True)
        self.allow_direct_oxidation = config.get("allow_direct_oxidation", False)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Look for bromination and hydrolysis reactions
        bromination_found = False
        hydrolysis_found = False
        direct_oxidation_found = False
        
        bromination_depth = -1
        hydrolysis_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzylic_bromination(rxn):
                bromination_found = True
                bromination_depth = i
            elif self.detect_benzylic_hydrolysis(rxn):
                hydrolysis_found = True
                hydrolysis_depth = i
            elif self.detect_direct_benzylic_oxidation(rxn):
                direct_oxidation_found = True
        
        # Check if we have the required pattern
        condition_met = False
        
        if self.require_sequential:
            # Must have both bromination and hydrolysis in correct order
            condition_met = (bromination_found and hydrolysis_found and 
                           bromination_depth < hydrolysis_depth)
        else:
            # Either sequential or direct oxidation allowed
            condition_met = ((bromination_found and hydrolysis_found) or 
                           (self.allow_direct_oxidation and direct_oxidation_found))
        
        return condition_met, len(reactions)
    
    def detect_benzylic_bromination(self, rxn):
        """Detect Wohl-Ziegler type benzylic bromination (C-H to C-Br)"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Pattern for benzylic methyl group
        benzylic_ch3_pattern = Chem.MolFromSmarts("[cH0:1]-[CH3:2]")
        # Pattern for benzylic bromide
        benzylic_br_pattern = Chem.MolFromSmarts("[cH0:1]-[CH2:2][Br]")
        
        # Check if product has benzylic CH3
        if not prod_mol or not prod_mol.HasSubstructMatch(benzylic_ch3_pattern):
            return False
            
        # Check if any reactant has benzylic bromide
        for react_mol in react_mols:
            if react_mol and react_mol.HasSubstructMatch(benzylic_br_pattern):
                return True
                
        # Also check for NBS (N-bromosuccinimide) or Br2 as brominating agent
        nbs_pattern = Chem.MolFromSmarts("O=C1CCC(=O)N1Br")
        br2_pattern = Chem.MolFromSmarts("BrBr")
        
        for react_mol in react_mols:
            if react_mol and (react_mol.HasSubstructMatch(nbs_pattern) or 
                             react_mol.HasSubstructMatch(br2_pattern)):
                return True
                
        return False
    
    def detect_benzylic_hydrolysis(self, rxn):
        """Detect hydrolysis of benzylic bromide to alcohol (C-Br to C-OH)"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Pattern for benzylic bromide
        benzylic_br_pattern = Chem.MolFromSmarts("[cH0:1]-[CH2:2][Br]")
        # Pattern for benzylic alcohol
        benzylic_oh_pattern = Chem.MolFromSmarts("[cH0:1]-[CH2:2][OH]")
        
        # Check if product has benzylic bromide
        has_benzylic_br_in_prod = prod_mol and prod_mol.HasSubstructMatch(benzylic_br_pattern)
        
        # Check if any reactant has benzylic alcohol
        has_benzylic_oh_in_react = False
        for react_mol in react_mols:
            if react_mol and react_mol.HasSubstructMatch(benzylic_oh_pattern):
                has_benzylic_oh_in_react = True
                break
                
        return has_benzylic_br_in_prod and has_benzylic_oh_in_react
    
    def detect_direct_benzylic_oxidation(self, rxn):
        """Detect direct oxidation of benzylic methyl to alcohol"""
        prod_mol, react_mols = self.parse_reaction(rxn)
        
        # Pattern for benzylic methyl group
        benzylic_ch3_pattern = Chem.MolFromSmarts("[cH0:1]-[CH3:2]")
        # Pattern for benzylic alcohol
        benzylic_oh_pattern = Chem.MolFromSmarts("[cH0:1]-[CH2:2][OH]")
        
        # Check if product has benzylic CH3
        has_benzylic_ch3_in_prod = prod_mol and prod_mol.HasSubstructMatch(benzylic_ch3_pattern)
        
        # Check if any reactant has benzylic alcohol
        has_benzylic_oh_in_react = False
        for react_mol in react_mols:
            if react_mol and react_mol.HasSubstructMatch(benzylic_oh_pattern):
                has_benzylic_oh_in_react = True
                break
                
        return has_benzylic_ch3_in_prod and has_benzylic_oh_in_react
    
    def parse_reaction(self, rxn):
        """Parse reaction SMILES to get product and reactant molecules"""
        try:
            rxn_parts = rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(rxn_parts[0])
            react_mols = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            return prod_mol, react_mols
        except:
            return None, []
