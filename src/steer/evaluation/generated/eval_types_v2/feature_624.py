"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where the target is assembled from 
    two major fragments via a coupling reaction (e.g., Suzuki coupling).
    Checks if the final step joins two substantial fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "suzuki").lower()
        self.final_step = config.get("final_step", True)
        self.min_fragment_size = config.get("min_fragment_size", 8)  # Minimum atoms per fragment
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling not found
        else:
            if self.final_step:
                # Reward convergent coupling as final step (x should be close to 0)
                return max(0, 10 - (x * 20))  # Penalize if not in final steps
            else:
                # General convergent coupling anywhere in route
                return 5  # Fixed reward for having convergent step
                
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent coupling of two fragments"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn[1].split(".")]
            product = Chem.MolFromSmiles(rxn[0])
            
            if not all([r is not None for r in reactants] + [product is not None]):
                return False
                
            # Check if we have exactly the expected number of major fragments
            major_fragments = [r for r in reactants if r.GetNumAtoms() >= self.min_fragment_size]
            
            if len(major_fragments) != self.fragment_count:
                return False
                
            # Check for coupling reaction patterns if specified
            if self.coupling_reaction == "suzuki":
                return self._is_suzuki_coupling(reactants, product)
            elif self.coupling_reaction == "buchwald":
                return self._is_buchwald_hartwig(reactants, product)
            elif self.coupling_reaction == "heck":
                return self._is_heck_coupling(reactants, product)
            else:
                # Generic C-C or C-N bond formation check
                return self._is_generic_coupling(major_fragments, product)
                
        except Exception:
            return False
            
    def _is_suzuki_coupling(self, reactants, product) -> bool:
        """Check for Suzuki coupling pattern: ArB(OR)2 + ArX -> Ar-Ar"""
        boronic_pattern = Chem.MolFromSmarts("[c,C]-B(-[O,OH])-[O,OH]")
        halide_pattern = Chem.MolFromSmarts("[c,C]-[Br,I,Cl]")
        
        has_boronic = any(r.HasSubstructMatch(boronic_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        return has_boronic and has_halide
        
    def _is_buchwald_hartwig(self, reactants, product) -> bool:
        """Check for Buchwald-Hartwig coupling: ArN + ArX -> Ar-N-Ar"""
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        halide_pattern = Chem.MolFromSmarts("[c,C]-[Br,I,Cl]")
        
        has_amine = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        return has_amine and has_halide
        
    def _is_heck_coupling(self, reactants, product) -> bool:
        """Check for Heck coupling: ArX + C=C -> Ar-C=C"""
        alkene_pattern = Chem.MolFromSmarts("C=C")
        halide_pattern = Chem.MolFromSmarts("[c,C]-[Br,I,Cl]")
        
        has_alkene = any(r.HasSubstructMatch(alkene_pattern) for r in reactants)
        has_halide = any(r.HasSubstructMatch(halide_pattern) for r in reactants)
        
        return has_alkene and has_halide
        
    def _is_generic_coupling(self, major_fragments, product) -> bool:
        """Check if major fragments are coupled together in product"""
        if len(major_fragments) < 2:
            return False
            
        # Simple heuristic: product should have significantly more atoms 
        # than individual fragments, indicating coupling
        total_reactant_atoms = sum(frag.GetNumAtoms() for frag in major_fragments)
        product_atoms = product.GetNumAtoms()
        
        # Allow for loss of small leaving groups (like HX, BOH, etc.)
        return abs(product_atoms - total_reactant_atoms) <= 10
