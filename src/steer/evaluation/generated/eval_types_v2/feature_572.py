"""Generated evaluation code for: Nitro to nitrile functional group interconversion sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitroToNitrileSequence(MultiRxnCondBase):
    """
    Evaluates routes for the presence of nitro to nitrile functional group 
    interconversion sequence (nitro reduction -> Sandmeyer reaction -> cyanation).
    
    Checks for consecutive occurrence of:
    1. Nitro group reduction (NO2 -> NH2)
    2. Sandmeyer reaction (NH2 -> I via diazonium)
    3. Cyanation (I -> CN)
    """
    
    def __init__(self, config):
        self.consecutive = config.get("consecutive", True)
        
        # SMARTS patterns for functional groups
        self.nitro_pattern = "[N+](=O)[O-]"
        self.amine_pattern = "[NH2]"
        self.iodide_pattern = "[I]"
        self.nitrile_pattern = "C#N"
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the nitro to nitrile sequence occurs in the route"""
        reactions = self.get_rxns(d)
        
        if len(reactions) < 3:  # Need at least 3 reactions for the sequence
            return False, len(reactions)
        
        if self.consecutive:
            condition = self._check_consecutive_sequence(reactions)
        else:
            condition = self._check_sequence_present(reactions)
        
        return condition, len(reactions)
    
    def _check_consecutive_sequence(self, reactions):
        """Check for consecutive nitro reduction -> Sandmeyer -> cyanation"""
        for i in range(len(reactions) - 2):
            if (self._is_nitro_reduction(reactions[i]) and
                self._is_sandmeyer_reaction(reactions[i + 1]) and
                self._is_cyanation(reactions[i + 2])):
                return True
        return False
    
    def _check_sequence_present(self, reactions):
        """Check if all three reaction types are present (not necessarily consecutive)"""
        has_nitro_reduction = any(self._is_nitro_reduction(r) for r in reactions)
        has_sandmeyer = any(self._is_sandmeyer_reaction(r) for r in reactions)
        has_cyanation = any(self._is_cyanation(r) for r in reactions)
        
        return has_nitro_reduction and has_sandmeyer and has_cyanation
    
    def _is_nitro_reduction(self, rxn):
        """Detect nitro group reduction: NO2 -> NH2"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = Chem.MolFromSmiles(rxn_parts[0])
            products_smiles = rxn_parts[1].split(".")
            products = [Chem.MolFromSmiles(p) for p in products_smiles if p]
            
            if not reactants or not products:
                return False
            
            # Check if reactant has nitro group and product has amine
            has_nitro_reactant = reactants.HasSubstructMatch(Chem.MolFromSmarts(self.nitro_pattern))
            has_amine_product = any(p.HasSubstructMatch(Chem.MolFromSmarts(self.amine_pattern)) for p in products)
            
            return has_nitro_reactant and has_amine_product
            
        except:
            return False
    
    def _is_sandmeyer_reaction(self, rxn):
        """Detect Sandmeyer reaction: NH2 -> I (via diazonium intermediate)"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            products_smiles = rxn_parts[1].split(".")
            products = [Chem.MolFromSmiles(p) for p in products_smiles if p]
            
            if not reactants or not products:
                return False
            
            # Check if reactant has amine and product has iodide
            has_amine_reactant = any(r.HasSubstructMatch(Chem.MolFromSmarts(self.amine_pattern)) for r in reactants)
            has_iodide_product = any(p.HasSubstructMatch(Chem.MolFromSmarts(self.iodide_pattern)) for p in products)
            
            return has_amine_reactant and has_iodide_product
            
        except:
            return False
    
    def _is_cyanation(self, rxn):
        """Detect cyanation: I -> CN"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            products_smiles = rxn_parts[1].split(".")
            products = [Chem.MolFromSmiles(p) for p in products_smiles if p]
            
            if not reactants or not products:
                return False
            
            # Check if reactant has iodide and product has nitrile
            has_iodide_reactant = any(r.HasSubstructMatch(Chem.MolFromSmarts(self.iodide_pattern)) for r in reactants)
            has_nitrile_product = any(p.HasSubstructMatch(Chem.MolFromSmarts(self.nitrile_pattern)) for p in products)
            
            return has_iodide_reactant and has_nitrile_product
            
        except:
            return False
