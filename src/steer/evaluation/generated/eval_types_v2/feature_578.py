"""Generated evaluation code for: Late isoxazole ring formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IsoxazoleCycloaddition(BaseScoring):
    """
    Evaluates synthesis routes for late-stage isoxazole ring formation via cycloaddition.
    Checks if an isoxazole ring (c1noc[cH]1) is formed through cycloaddition reactions
    and rewards earlier occurrence in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen via cycloaddition
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage formation is rewarded (lower depth fraction is better)
            else:
                return x  # Early-stage formation is rewarded (higher depth fraction is better)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an isoxazole ring via cycloaddition"""
        try:
            # Get the mapped reaction SMILES
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if isoxazole ring is formed (present in products but not in reactants)
            has_isoxazole_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            has_isoxazole_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            if not (has_isoxazole_in_products and not has_isoxazole_in_reactants):
                return False
            
            # Check if it's a cycloaddition reaction (typical patterns)
            if self.formation_method == "cycloaddition":
                return self._is_cycloaddition_reaction(reactants, products)
            
            return True
            
        except Exception:
            return False
    
    def _is_cycloaddition_reaction(self, reactants, products) -> bool:
        """Check if the reaction follows cycloaddition patterns"""
        # For isoxazole formation, typical [3+2] cycloaddition involves:
        # - A dipole (like nitrile oxide) and a dipolarophile (like alkene/alkyne)
        # - Usually 2 reactants combining to form 1 main product
        
        if len(reactants) != 2:
            return False
        
        # Check for typical cycloaddition fingerprints:
        # 1. Nitrile oxide pattern (C#N-O or N=O functional groups)
        # 2. Alkene/alkyne dipolarophile pattern
        nitrile_oxide_pattern = Chem.MolFromSmarts("[C]#[N+][O-]")  # Nitrile oxide
        alkyne_pattern = Chem.MolFromSmarts("C#C")  # Terminal or internal alkyne
        alkene_pattern = Chem.MolFromSmarts("C=C")  # Alkene
        
        has_dipole = any(r.HasSubstructMatch(nitrile_oxide_pattern) for r in reactants if r)
        has_dipolarophile = any(r.HasSubstructMatch(alkyne_pattern) or r.HasSubstructMatch(alkene_pattern) 
                               for r in reactants if r)
        
        return has_dipole and has_dipolarophile
