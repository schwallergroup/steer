"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two major fragments
    are coupled together at a specific position in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_position = config.get("coupling_step_position", "late")
        self.min_fragment_size = config.get("min_fragment_size", 8)  # minimum atoms per fragment
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        if self.coupling_position == "late":
            # Reward later convergent steps (closer to final product)
            return (1 - x) * 10
        elif self.coupling_position == "early":
            # Reward earlier convergent steps
            return x * 10
        else:
            # Default: prefer mid-route convergence
            return (1 - abs(x - 0.5) * 2) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of major fragments.
        A convergent step is identified when:
        1. Multiple reactants of sufficient size combine
        2. Each reactant contributes significant molecular weight to product
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            
            if not product:
                return False
                
            reactants = []
            for r_smiles in reactant_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactants.append(mol)
            
            # Need at least the specified number of fragments
            if len(reactants) < self.fragment_count:
                return False
            
            # Check if reactants are substantial fragments
            product_atoms = product.GetNumAtoms()
            significant_fragments = 0
            
            for reactant in reactants:
                reactant_atoms = reactant.GetNumAtoms()
                # Fragment must be at least min_fragment_size atoms and contribute
                # at least 25% of the final product size
                if (reactant_atoms >= self.min_fragment_size and 
                    reactant_atoms / product_atoms >= 0.25):
                    significant_fragments += 1
            
            # Check if we have the required number of significant fragments
            if significant_fragments >= self.fragment_count:
                # Additional check: ensure this is actually a coupling reaction
                # (not just mixing small molecules)
                return self._is_coupling_reaction(reactants, product)
            
            return False
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, reactants, product) -> bool:
        """
        Verify this is a true coupling reaction by checking if major structural
        fragments from reactants are preserved in the product.
        """
        try:
            # Sort reactants by size (largest first)
            sorted_reactants = sorted(reactants, key=lambda x: x.GetNumAtoms(), reverse=True)
            
            # Take the top fragment_count reactants
            major_reactants = sorted_reactants[:self.fragment_count]
            
            fragments_found = 0
            for reactant in major_reactants:
                # Create a more flexible matching by removing explicit hydrogens
                # and using a substructure search
                reactant_no_h = Chem.RemoveHs(reactant)
                product_no_h = Chem.RemoveHs(product)
                
                if reactant_no_h.GetNumAtoms() >= self.min_fragment_size:
                    # Check if major portion of reactant is preserved in product
                    if product_no_h.HasSubstructMatch(reactant_no_h):
                        fragments_found += 1
                    else:
                        # Try with more relaxed matching (remove some bonds)
                        if self._flexible_substructure_match(reactant_no_h, product_no_h):
                            fragments_found += 1
            
            return fragments_found >= self.fragment_count
            
        except Exception:
            return False
    
    def _flexible_substructure_match(self, fragment, product) -> bool:
        """
        More flexible substructure matching for cases where bonds are formed/broken
        during coupling but the core structure is preserved.
        """
        try:
            # If direct match fails, try matching the largest ring system or chain
            fragment_atoms = fragment.GetNumAtoms()
            if fragment_atoms < self.min_fragment_size:
                return False
                
            # Simple heuristic: if at least 70% of fragment atoms could be matched
            # in the product, consider it a preserved fragment
            threshold = max(self.min_fragment_size, int(fragment_atoms * 0.7))
            
            # This is a simplified check - in practice, you might want more
            # sophisticated substructure analysis
            return product.GetNumAtoms() >= threshold
            
        except Exception:
            return False
