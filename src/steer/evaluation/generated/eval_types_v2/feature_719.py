"""Generated evaluation code for: Late stage Suzuki coupling for aryl-pyridine bond"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiArylPyridine(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction forming an aryl-pyridine bond 
    occurs late in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
        # SMARTS patterns for detecting aryl-pyridine bond formation
        self.pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")  # Pyridine ring
        self.aryl_pattern = Chem.MolFromSmarts("c1ccccc1")  # Aromatic ring
        self.aryl_pyridine_pattern = Chem.MolFromSmarts("c1ccccc1-c1ccncc1")  # Aryl-pyridine bond

    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x >= 0.7 else 0  # Late stage threshold
        else:
            # Prefer reactions closer to target depth (late stage)
            return 10 * max(0, 1 - abs(x - self.target_depth))

    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling forming aryl-pyridine bond"""
        metadata = d.get("metadata", {})
        
        # Check if it's identified as Suzuki coupling
        policy_name = metadata.get("policy_name", "")
        if "suzuki" not in policy_name.lower():
            # Also check reaction template or classification if available
            reaction_type = metadata.get("reaction_type", "")
            if "suzuki" not in reaction_type.lower() and "coupling" not in reaction_type.lower():
                return False
        
        # Get mapped reaction SMILES
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains aryl-pyridine bond
            if not product.HasSubstructMatch(self.aryl_pyridine_pattern):
                return False
            
            # Verify bond formation: aryl-pyridine bond should be present in product 
            # but the connected components should be separate in reactants
            return self._verify_aryl_pyridine_formation(product, reactants)
            
        except Exception:
            return False

    def _verify_aryl_pyridine_formation(self, product, reactants) -> bool:
        """Verify that aryl-pyridine bond is actually formed in this step"""
        # Find aryl-pyridine matches in product
        matches = product.GetSubstructMatches(self.aryl_pyridine_pattern)
        if not matches:
            return False
        
        # Check if we have separate aryl and pyridine containing reactants
        has_aryl_reactant = False
        has_pyridine_reactant = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.pyridine_pattern):
                has_pyridine_reactant = True
            if reactant.HasSubstructMatch(self.aryl_pattern):
                has_aryl_reactant = True
        
        # Should have both aryl and pyridine reactants for Suzuki coupling
        if not (has_aryl_reactant and has_pyridine_reactant):
            return False
        
        # Check that aryl-pyridine bond is not present in any single reactant
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.aryl_pyridine_pattern):
                return False  # Bond already exists, not formed in this step
        
        return True
