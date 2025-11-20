"""Generated evaluation code for: Early triazolo-pyridazinone core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriazoloPyridazinoneCore(BaseScoring):
    """
    Evaluates early triazolo-pyridazinone core construction.
    
    Checks if triazole or pyridazinone ring patterns are formed early in the synthesis
    via condensation reactions, rewarding construction before the depth threshold.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.depth_threshold = config["parameters"]["depth_threshold"]
        self.formation_type = config["parameters"]["formation_type"]
        
        # Compile SMARTS patterns for efficiency
        self.ring_patterns = [Chem.MolFromSmarts(smarts) for smarts in self.ring_smarts]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        # Convert depth to fraction (0 = root, 1 = leaves)
        depth_fraction = x
        
        # Reward early formation (before threshold)
        if depth_fraction <= (self.depth_threshold / 10.0):  # Normalize threshold
            return 10.0 * (1.0 - depth_fraction)  # Higher score for earlier formation
        else:
            return 2.0 * (1.0 - depth_fraction)  # Lower reward for late formation
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a triazole or pyridazinone ring via condensation."""
        metadata = d.get("metadata", {})
        
        # Check if it's a condensation reaction
        policy_name = metadata.get("policy_name", "")
        if self.formation_type == "condensation" and "condensation" not in policy_name.lower():
            return False
        
        # Get reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains target ring patterns
            product_has_ring = any(product_mol.HasSubstructMatch(pattern) 
                                 for pattern in self.ring_patterns if pattern)
            
            if not product_has_ring:
                return False
            
            # Check if reactants lack the ring patterns (ring formation)
            reactants_have_ring = any(
                any(reactant.HasSubstructMatch(pattern) for pattern in self.ring_patterns if pattern)
                for reactant in reactant_mols
            )
            
            # Ring formation occurs if product has ring but reactants don't
            return product_has_ring and not reactants_have_ring
            
        except Exception:
            return False
