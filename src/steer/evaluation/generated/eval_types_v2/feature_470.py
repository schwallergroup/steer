"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ring formation.
    Checks if a specified ring pattern is formed late in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Late-stage formation is preferred (higher depth fraction is better)
            return x * 10
        elif self.timing == "early":
            # Early-stage formation is preferred (lower depth fraction is better)
            return (1 - x) * 10
        else:
            # Default to late-stage preference
            return x * 10
    
    def hit_condition(self, d):
        """
        Check if this reaction involves formation of the target ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            # Parse products and reactants
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains the ring pattern
            product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if self.direction == "formation":
                # Ring formation: product has ring, but no reactant has the complete ring
                if not product_has_ring:
                    return False
                
                # Check if any reactant already has the complete ring
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
                
                # Ring formation occurs if product has ring but reactants don't
                return not reactants_have_ring
                
            elif self.direction == "breaking":
                # Ring breaking: reactants have ring, but product doesn't
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
                return reactants_have_ring and not product_has_ring
            
        except Exception:
            return False
            
        return False
