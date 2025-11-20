"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are joined at a specific stage.
    Checks if the route assembles the target via coupling of exactly two fragments
    at the specified coupling stage (e.g., final step).
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "final")
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0  # Positive if convergent coupling found
        else:
            if x < 0:
                return 0  # No convergent coupling found
            return max(0, 1 - abs(x - self.target_depth))  # Closer to target depth is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling of fragments."""
        try:
            # Get mapped reaction SMILES
            mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            product_smiles = rxn_parts[0].strip()
            reactants_smiles = rxn_parts[1].strip()
            
            # Parse reactants
            reactant_smiles_list = [r.strip() for r in reactants_smiles.split(".")]
            
            # For convergent synthesis, we expect exactly the specified number of fragments
            if len(reactant_smiles_list) != self.fragment_count:
                return False
            
            # Convert to RDKit molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            
            if not product or None in reactants:
                return False
            
            # Check if this is a true coupling reaction (fragments combine to form product)
            # Verify that each reactant contributes substantial atoms to the product
            product_atom_count = product.GetNumAtoms()
            min_contribution = max(3, product_atom_count * 0.2)  # At least 20% or 3 atoms
            
            for reactant in reactants:
                reactant_atom_count = reactant.GetNumAtoms()
                if reactant_atom_count < min_contribution:
                    return False
            
            # Check that the sum of reactant atoms is reasonable compared to product
            # (accounting for potential small leaving groups or added atoms)
            total_reactant_atoms = sum(r.GetNumAtoms() for r in reactants)
            if abs(total_reactant_atoms - product_atom_count) > product_atom_count * 0.3:
                return False
            
            # Additional check: ensure this isn't just a simple functional group transformation
            # by verifying that both fragments have reasonable complexity
            for reactant in reactants:
                if reactant.GetNumAtoms() < 5:  # Very small fragments are likely reagents
                    return False
            
            return True
            
        except Exception:
            return False
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Override to handle specific coupling stage requirements."""
        if self.coupling_stage == "final":
            # For final stage coupling, only check the root reaction
            if self.hit_condition(d):
                return True, 0
            return False, -1
        else:
            # Use default BFS behavior for other stages
            return super().condition_depth(d)
