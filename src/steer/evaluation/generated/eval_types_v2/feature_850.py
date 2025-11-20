"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two substantial 
    fragments are coupled together to form the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")
        self.min_heavy_atoms = config.get("min_heavy_atoms_per_fragment", 8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            if self.coupling_step == "final":
                # Reward early convergence (lower depth)
                return 1 - x
            else:
                # Flexible scoring based on depth
                return max(0, 1 - abs(x - 0.5))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Need at least the specified number of reactant fragments
            if len(reactants_smiles) < self.fragment_count:
                return False
                
            # Convert to molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            reactants = [r for r in reactants if r is not None]
            
            if not product or len(reactants) < self.fragment_count:
                return False
                
            # Check that we have substantial fragments (not just small coupling reagents)
            substantial_reactants = []
            for reactant in reactants:
                heavy_atom_count = reactant.GetNumHeavyAtoms()
                if heavy_atom_count >= self.min_heavy_atoms:
                    substantial_reactants.append(reactant)
            
            # Must have at least the required number of substantial fragments
            if len(substantial_reactants) < self.fragment_count:
                return False
                
            # Check that the fragments are being coupled (not just functionalized)
            # This is indicated by the product having significantly more heavy atoms
            # than any individual reactant
            product_heavy_atoms = product.GetNumHeavyAtoms()
            max_reactant_heavy_atoms = max(r.GetNumHeavyAtoms() for r in substantial_reactants)
            
            # The product should be substantially larger than the largest fragment
            if product_heavy_atoms < max_reactant_heavy_atoms + self.min_heavy_atoms // 2:
                return False
                
            # Additional check: ensure fragments contribute roughly equally
            # (to distinguish from minor modifications)
            sorted_sizes = sorted([r.GetNumHeavyAtoms() for r in substantial_reactants], reverse=True)
            if len(sorted_sizes) >= 2:
                size_ratio = sorted_sizes[1] / sorted_sizes[0]
                if size_ratio < 0.3:  # Second largest fragment should be at least 30% of largest
                    return False
                    
            return True
            
        except Exception:
            return False
