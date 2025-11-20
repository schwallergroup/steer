"""Generated evaluation code for: Convergent cycloaddition for bicyclic core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentCycloaddition(BaseScoring):
    """
    Evaluates routes for convergent cycloaddition reactions that build bicyclic cores.
    Detects [3+2], [4+2], and other cycloaddition patterns that create bicyclic systems
    from two or more fragments in a single step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "linear")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
        self.min_fragments = config.get("fragments_count", 2)
        
        # Common bicyclic core patterns
        self.bicyclic_patterns = [
            "[#6]1~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]12",  # bicyclo[3.2.1]octane
            "[#6]1~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]12",        # bicyclo[2.2.1]heptane
            "[#6]1~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]12",        # bicyclo[3.1.1]heptane
            "[#6]1~[#6]~[#6]~[#6]2~[#6]~[#6]12",             # bicyclo[2.1.1]hexane
            "[#6]1~[#6]2~[#6]~[#6]~[#6]~[#6]12",             # bicyclo[1.2.1]hexane
        ]

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 10 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            # Earlier cycloaddition is better (more convergent)
            return 10 * (1 - abs(x - self.target_depth))

    def hit_condition(self, d) -> bool:
        """Check if this reaction is a convergent cycloaddition forming a bicyclic core."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles.strip())
            
            if not all(reactants) or not product:
                return False
                
            # Check if we have at least the minimum number of fragments
            if len(reactants) < self.min_fragments:
                return False
                
            # Check if product contains a bicyclic core
            has_bicyclic_core = any(
                product.HasSubstructMatch(Chem.MolFromSmarts(pattern))
                for pattern in self.bicyclic_patterns
            )
            
            if not has_bicyclic_core:
                return False
                
            # Check if this is likely a cycloaddition by analyzing ring formation
            product_ring_info = product.GetRingInfo()
            product_rings = len(product_ring_info.AtomRings())
            
            # Count rings in reactants
            total_reactant_rings = sum(
                len(mol.GetRingInfo().AtomRings()) for mol in reactants
            )
            
            # Cycloaddition should form new rings (at least 1-2 new rings)
            new_rings_formed = product_rings - total_reactant_rings
            if new_rings_formed < 1:
                return False
                
            # Check for convergent nature - reactants should be of similar complexity
            reactant_sizes = [mol.GetNumAtoms() for mol in reactants]
            if len(reactant_sizes) >= 2:
                # Calculate complexity balance (avoid one very small fragment)
                min_size = min(reactant_sizes)
                max_size = max(reactant_sizes)
                if min_size < 3 or max_size / min_size > 5:  # Too imbalanced
                    return False
                    
            return True
            
        except Exception:
            return False
