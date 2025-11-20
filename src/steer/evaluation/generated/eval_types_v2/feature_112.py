"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentUreaSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy via two complex fragments joined by urea formation.
    Checks for late-stage urea coupling between two substantial molecular fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "urea_formation")
        self.timing = config.get("timing", "late")
        
        # SMARTS pattern for urea formation (C(=O)N-N, amine + isocyanate, etc.)
        self.urea_pattern = Chem.MolFromSmarts("[C](=O)([NX3])[NX3]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent urea coupling doesn't occur
        
        if self.timing == "late":
            # Reward later convergent coupling (lower depth fraction is better)
            return max(0, 10 * (1 - x))
        else:
            # Neutral timing preference
            return 5 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent urea formation between complex fragments"""
        try:
            metadata = d.get("metadata", {})
            mapped_rxn = metadata.get("mapped_reaction_smiles", "")
            
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
                
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            if "." not in reactants_smiles:
                return False  # Need multiple reactants for convergent synthesis
                
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or len(reactants) < 2:
                return False
                
            # Check if product contains urea and reactants don't
            product_has_urea = product.HasSubstructMatch(self.urea_pattern)
            reactants_have_urea = any(r.HasSubstructMatch(self.urea_pattern) for r in reactants if r)
            
            if not (product_has_urea and not reactants_have_urea):
                return False
                
            # Check for convergent coupling: two substantial fragments
            substantial_fragments = []
            for reactant in reactants:
                if reactant and self._is_substantial_fragment(reactant):
                    substantial_fragments.append(reactant)
                    
            # Must have exactly the required number of substantial fragments
            return len(substantial_fragments) == self.fragment_count
            
        except Exception:
            return False
    
    def _is_substantial_fragment(self, mol) -> bool:
        """Determine if a molecule qualifies as a substantial/complex fragment"""
        if not mol:
            return False
            
        # Criteria for substantial fragment:
        # - At least 8 heavy atoms
        # - Contains at least one ring or heteroatom
        # - Not just a simple coupling reagent
        
        heavy_atom_count = mol.GetNumHeavyAtoms()
        if heavy_atom_count < 8:
            return False
            
        # Check for structural complexity
        ring_info = mol.GetRingInfo()
        has_rings = ring_info.NumRings() > 0
        
        heteroatoms = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetAtomicNum() not in [1, 6])  # Not H or C
        has_heteroatoms = heteroatoms > 1
        
        # Simple coupling reagents to exclude (isocyanates, simple amines)
        simple_isocyanate = Chem.MolFromSmarts("[C](=O)=N")
        simple_amine = Chem.MolFromSmarts("[CH2,CH3][NH2]")
        
        is_simple_reagent = (mol.HasSubstructMatch(simple_isocyanate) and heavy_atom_count < 12) or \
                           (mol.HasSubstructMatch(simple_amine) and heavy_atom_count < 6)
        
        return (has_rings or has_heteroatoms) and not is_simple_reagent
