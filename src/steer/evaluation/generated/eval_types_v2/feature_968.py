"""Generated evaluation code for: Late stage cyclopropane ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates routes based on late-stage formation of specific ring systems.
    Rewards routes where the target ring is formed closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For late-stage formation, lower depth fractions (closer to end) are better.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            # Reward formations closer to the final step (lower x values)
            return (1 - x) * 10
        elif self.timing == "early":
            # Reward formations closer to the starting materials (higher x values)
            return x * 10
        else:
            # Default case - any formation gets partial credit
            return 5
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves formation of the target ring.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            # Parse reactants and product
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check if ring formation occurred
            if self.direction == "formation":
                # Product should have the ring
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                # At least one reactant should NOT have the ring
                reactants_have_ring = [mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols if mol]
                
                # Ring formation: product has ring, but not all reactants have it
                return product_has_ring and not all(reactants_have_ring)
            
            elif self.direction == "breaking":
                # At least one reactant should have the ring
                reactants_have_ring = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols if mol)
                
                # Product should NOT have the ring
                product_has_ring = product_mol.HasSubstructMatch(self.ring_pattern)
                
                # Ring breaking: reactant has ring, product doesn't
                return reactants_have_ring and not product_has_ring
            
        except Exception:
            return False
        
        return False
