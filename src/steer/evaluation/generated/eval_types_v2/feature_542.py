"""Generated evaluation code for: Acetate protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AcetateCyclingStrategy(MultiRxnCondBase):
    """
    Evaluates acetate protecting group cycling strategy where an acetate group
    is installed to protect an alcohol and then removed after selective functionalization.
    Looks for the pattern: acetate installation -> intermediate reactions -> acetate removal.
    """
    
    def __init__(self, config):
        self.min_cycle_length = config.get("min_cycle_length", 2)
        self.max_cycle_length = config.get("max_cycle_length", 5)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find acetate installation and removal reactions
        acetate_installations = []
        acetate_removals = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_acetate_installation(rxn):
                acetate_installations.append(i)
            elif self.detect_acetate_removal(rxn):
                acetate_removals.append(i)
        
        # Check for cycling pattern
        cycling_found = self.detect_cycling_pattern(acetate_installations, acetate_removals)
        
        return cycling_found, len(reactions)
    
    def detect_acetate_installation(self, rxn):
        """Detect installation of acetate protecting group on alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Look for acetyl reagent in reactants (acetyl chloride, acetic anhydride, etc.)
        acetyl_patterns = [
            "CC(=O)Cl",  # acetyl chloride
            "CC(=O)OC(=O)C",  # acetic anhydride
            "CC(=O)O"  # acetic acid (with coupling agent)
        ]
        
        has_acetyl_reagent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in acetyl_patterns)
            for mol in reactants
        )
        
        # Look for alcohol in reactants and acetate ester in products
        alcohol_pattern = "[OH1]"  # primary or secondary alcohol
        acetate_pattern = "CC(=O)O"  # acetate ester
        
        has_alcohol_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(alcohol_pattern))
            for mol in reactants
        )
        
        has_acetate_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(acetate_pattern))
            for mol in products
        )
        
        return has_acetyl_reagent and has_alcohol_reactant and has_acetate_product
    
    def detect_acetate_removal(self, rxn):
        """Detect removal/deprotection of acetate group to regenerate alcohol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".") if r.strip()]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".") if p.strip()]
        
        if not all(reactants) or not all(products):
            return False
        
        # Look for acetate ester in reactants
        acetate_pattern = "CC(=O)O"
        has_acetate_reactant = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(acetate_pattern))
            for mol in reactants
        )
        
        # Look for alcohol in products
        alcohol_pattern = "[OH1]"
        has_alcohol_product = any(
            mol.HasSubstructMatch(Chem.MolFromSmarts(alcohol_pattern))
            for mol in products
        )
        
        # Look for typical deprotection conditions (base, acid, or enzymatic)
        deprotection_patterns = [
            "[OH-]",  # hydroxide
            "[O-]C",  # methoxide
            "N",  # amine base
        ]
        
        has_deprotection_reagent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in deprotection_patterns)
            for mol in reactants
        )
        
        return has_acetate_reactant and has_alcohol_product
    
    def detect_cycling_pattern(self, installations, removals):
        """Check if there's a valid cycling pattern with appropriate spacing"""
        if not installations or not removals:
            return False
        
        # Check for installation followed by removal within the cycle length range
        for install_idx in installations:
            for remove_idx in removals:
                if install_idx < remove_idx:  # Installation must come before removal
                    cycle_length = remove_idx - install_idx
                    if self.min_cycle_length <= cycle_length <= self.max_cycle_length:
                        return True
        
        return False
    
    def route_scoring(self, x):
        """Score based on presence of cycling strategy"""
        if x < 0:
            return 0  # No cycling pattern found
        else:
            return 1 - x  # Earlier detection (shorter routes) scored higher
