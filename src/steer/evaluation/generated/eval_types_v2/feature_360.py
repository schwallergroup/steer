"""Generated evaluation code for: Cyclopropanation via carbene addition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CyclopropanationViaCarbene(BaseScoring):
    """
    Evaluates synthesis routes for cyclopropanation reactions via carbene addition.
    Detects the formation of cyclopropane rings through carbene intermediates,
    particularly from diazomethane or similar carbene precursors.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for carbene precursors (diazomethane pattern)
            carbene_precursor_pattern = Chem.MolFromSmarts("[C-]=[N+]=[N-]")  # Diazomethane
            carbene_precursor_pattern2 = Chem.MolFromSmarts("C=[N+]=[N-]")    # Alternative form
            
            has_carbene_precursor = any(
                mol.HasSubstructMatch(carbene_precursor_pattern) or 
                mol.HasSubstructMatch(carbene_precursor_pattern2)
                for mol in reactants if mol is not None
            )
            
            # Check for cyclopropane formation
            cyclopropane_pattern = Chem.MolFromSmarts("C1CC1")  # Three-membered carbon ring
            
            # Count cyclopropane rings in reactants vs products
            reactant_cyclopropanes = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern))
                for mol in reactants if mol is not None
            )
            
            product_cyclopropanes = sum(
                len(mol.GetSubstructMatches(cyclopropane_pattern))
                for mol in products if mol is not None
            )
            
            # Cyclopropanation occurred if we have more cyclopropanes in products
            cyclopropane_formed = product_cyclopropanes > reactant_cyclopropanes
            
            # Additional check for alkene substrate (common in cyclopropanation)
            alkene_pattern = Chem.MolFromSmarts("C=C")
            has_alkene_substrate = any(
                mol.HasSubstructMatch(alkene_pattern)
                for mol in reactants if mol is not None
            )
            
            return (has_carbene_precursor or 
                   d.get("metadata", {}).get("policy_name") == "carbene_cyclopropanation") and \
                   cyclopropane_formed and has_alkene_substrate
                   
        except Exception:
            return False
