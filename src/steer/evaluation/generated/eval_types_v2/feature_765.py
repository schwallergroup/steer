"""Generated evaluation code for: Multiple symmetric substrate mono-functionalization steps"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiSymmetricMonoFunctionalization(MultiRxnCondBase):
    """
    Evaluates routes containing multiple mono-functionalization steps of symmetric 
    difunctional substrates (diboronic acids, bis-boronic esters).
    
    Checks if the route contains at least the specified count of reactions where
    a symmetric difunctional substrate is selectively mono-functionalized.
    """
    
    def __init__(self, config):
        self.required_count = config["parameters"]["count"]
        self.target_substrates = config["parameters"]["substrates"]
        
        # SMARTS patterns for symmetric difunctional substrates
        self.substrate_patterns = {
            "diboronic_acid": "[#6]-B(O)O.*[#6]-B(O)O",
            "bis_boronic_ester": "[#6]-B1OC([CH3])([CH3])C([CH3])([CH3])O1.*[#6]-B1OC([CH3])([CH3])C([CH3])([CH3])O1"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        mono_functionalization_count = 0
        
        for rxn in reactions:
            if self.detect_mono_functionalization(rxn):
                mono_functionalization_count += 1
        
        condition_met = mono_functionalization_count >= self.required_count
        return condition_met, len(reactions)
    
    def detect_mono_functionalization(self, rxn) -> bool:
        """
        Detects if a reaction involves mono-functionalization of a symmetric difunctional substrate.
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check each reactant for symmetric difunctional patterns
        for reactant_smiles in reactants:
            try:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol is None:
                    continue
                    
                # Check if reactant matches any target substrate pattern
                has_target_substrate = False
                for substrate_type in self.target_substrates:
                    if substrate_type in self.substrate_patterns:
                        pattern = Chem.MolFromSmarts(self.substrate_patterns[substrate_type])
                        if pattern and reactant_mol.HasSubstructMatch(pattern):
                            has_target_substrate = True
                            break
                
                if not has_target_substrate:
                    continue
                
                # Check if one functional group remains in products (mono-functionalization)
                if self.check_mono_selectivity(reactant_mol, products, substrate_type):
                    return True
                    
            except Exception:
                continue
        
        return False
    
    def check_mono_selectivity(self, reactant_mol, products, substrate_type) -> bool:
        """
        Verifies that exactly one functional group was consumed, leaving one intact.
        """
        pattern = Chem.MolFromSmarts(self.substrate_patterns[substrate_type])
        if not pattern:
            return False
            
        # Count functional groups in reactant
        reactant_matches = len(reactant_mol.GetSubstructMatches(pattern))
        if reactant_matches == 0:
            return False
        
        # Check products for remaining functional groups
        for product_smiles in products:
            try:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue
                    
                # Look for single functional group pattern
                single_group_pattern = self.get_single_group_pattern(substrate_type)
                if single_group_pattern:
                    single_pattern_mol = Chem.MolFromSmarts(single_group_pattern)
                    if single_pattern_mol and product_mol.HasSubstructMatch(single_pattern_mol):
                        # Ensure it's not still symmetric (should have exactly 1 match)
                        single_matches = len(product_mol.GetSubstructMatches(single_pattern_mol))
                        if single_matches == 1:
                            return True
                            
            except Exception:
                continue
        
        return False
    
    def get_single_group_pattern(self, substrate_type) -> str:
        """
        Returns SMARTS pattern for single functional group of the substrate type.
        """
        single_patterns = {
            "diboronic_acid": "[#6]-B(O)O",
            "bis_boronic_ester": "[#6]-B1OC([CH3])([CH3])C([CH3])([CH3])O1"
        }
        return single_patterns.get(substrate_type, "")
