"""Generated evaluation code for: Convergent ether coupling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentEtherCoupling(BaseScoring):
    """
    Evaluates convergent ether coupling strategy where two major molecular fragments 
    are coupled via Williamson ether synthesis at a specific depth.
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["parameters"]["depth"]
        self.min_fragment_count = config["parameters"]["fragment_count"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        else:
            # Perfect score if at target depth, penalty for deviation
            depth_penalty = abs(x - self.target_depth) * 0.2
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent Williamson ether synthesis"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        if "." not in reactants_smiles:
            return False  # Need multiple reactants for convergent synthesis
            
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or len(reactants) < self.min_fragment_count:
                return False
                
            # Check if this is an ether formation reaction
            if not self._is_ether_formation(product, reactants):
                return False
                
            # Check if reactants are substantial fragments (convergent criterion)
            return self._are_convergent_fragments(reactants)
            
        except Exception:
            return False
    
    def _is_ether_formation(self, product, reactants) -> bool:
        """Check if the reaction forms an ether bond"""
        # Look for ether oxygen in product that connects fragments
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        if not product.HasSubstructMatch(ether_pattern):
            return False
            
        # Check for typical Williamson ether synthesis patterns
        # Alkyl halide + alkoxide -> ether + halide salt
        halide_pattern = Chem.MolFromSmarts("[C][Cl,Br,I]")
        alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
        
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        has_alcohol = any(r.HasSubstructMatch(alcohol_pattern) for r in reactants)
        
        return has_halide and has_alcohol
    
    def _are_convergent_fragments(self, reactants) -> bool:
        """Check if reactants represent substantial molecular fragments"""
        # Filter out small molecules (salts, solvents, etc.)
        substantial_fragments = []
        
        for reactant in reactants:
            if reactant.GetNumHeavyAtoms() >= 6:  # Minimum size for substantial fragment
                # Exclude simple salts and common reagents
                salt_patterns = [
                    "[Na,K,Li]",
                    "[Cl,Br,I]-",
                    "O=S(=O)([O-])[O-]"  # sulfate
                ]
                
                is_salt = any(reactant.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                             for p in salt_patterns)
                
                if not is_salt:
                    substantial_fragments.append(reactant)
        
        # Need at least the specified number of substantial fragments
        return len(substantial_fragments) >= self.min_fragment_count
