"""Generated evaluation code for: Late stage cyclopropanation of allyl group"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAllyicCyclopropanation(BaseScoring):
    """
    Evaluates late-stage cyclopropanation reactions that convert allyl groups to cyclopropyl groups.
    
    Detects reactions where an allyl group (C=C-C) is transformed into a cyclopropyl group,
    with preference for reactions occurring later in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclopropanation doesn't occur
        else:
            if self.timing_preference == "late":
                return 1 - x  # Later stages get higher scores
            else:
                return 1  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves cyclopropanation of an allyl group"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            prod_mol = Chem.MolFromSmiles(prod_smiles)
            react_mols = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
            
            # Define patterns
            allyl_pattern = Chem.MolFromSmarts("[*]-C-C=C")  # Allyl group attached to something
            cyclopropyl_pattern = Chem.MolFromSmarts("[*]-C1CC1")  # Cyclopropyl group attached to something
            
            if not allyl_pattern or not cyclopropyl_pattern:
                return False
            
            # Check if product has cyclopropyl and reactants have allyl
            has_cyclopropyl_product = prod_mol.HasSubstructMatch(cyclopropyl_pattern)
            has_allyl_reactant = any(mol.HasSubstructMatch(allyl_pattern) for mol in react_mols)
            
            # Additional check: ensure we're not just seeing pre-existing cyclopropyl groups
            has_cyclopropyl_reactant = any(mol.HasSubstructMatch(cyclopropyl_pattern) for mol in react_mols)
            
            # True cyclopropanation: allyl in reactants -> cyclopropyl in products (without pre-existing cyclopropyl)
            return (has_cyclopropyl_product and has_allyl_reactant and 
                   (not has_cyclopropyl_reactant or self._net_cyclopropyl_increase(prod_mol, react_mols, cyclopropyl_pattern)))
            
        except Exception:
            return False
    
    def _net_cyclopropyl_increase(self, prod_mol, react_mols, cyclopropyl_pattern) -> bool:
        """Check if there's a net increase in cyclopropyl groups"""
        prod_count = len(prod_mol.GetSubstructMatches(cyclopropyl_pattern))
        react_count = sum(len(mol.GetSubstructMatches(cyclopropyl_pattern)) for mol in react_mols)
        return prod_count > react_count
