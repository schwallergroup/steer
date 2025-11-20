"""Generated evaluation code for: Convergent assembly via Suzuki coupling of fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiAssembly(BaseScoring):
    """
    Evaluates convergent assembly strategy using Suzuki coupling to join fragments.
    Checks for late-stage Suzuki coupling reactions that connect two major fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.timing_preference = config.get("timing", "late_stage")  # late_stage, mid_stage, early_stage
        
        # Suzuki coupling SMARTS patterns
        self.suzuki_patterns = [
            # Aryl-aryl coupling products (biaryl formation)
            "[c:1]-[c:2]",
            # Vinyl-aryl coupling products  
            "[C:1]=[C:2]-[c:3]",
            # General C-C bond formation between sp2 carbons
            "[$(c),$(C=C):1]-[$(c),$(C=C):2]"
        ]
        
        # Boronic acid/ester patterns (reactant side)
        self.boronic_patterns = [
            "[c,C]-B(O)(O)",  # Boronic acid
            "[c,C]-B1OCC(C)(C)CO1",  # Pinacol boronate
            "[c,C]-B(O[C,c])(O[C,c])"  # General boronic ester
        ]
        
        # Halide patterns (other reactant)
        self.halide_patterns = [
            "[c,C]-[Br,I,Cl]"
        ]

    def route_scoring(self, x: float) -> float:
        """Convert depth fraction to score (0-10)."""
        if x < 0:
            return 0  # Suzuki coupling not found
        
        # For late-stage timing preference, lower depth fraction is better
        if self.timing_preference == "late_stage":
            return (1 - x) * 10  # Late stage gets higher score
        elif self.timing_preference == "mid_stage":
            # Peak score around 0.5 depth fraction
            return 10 * (1 - 4 * (x - 0.5) ** 2) if 0.25 <= x <= 0.75 else 0
        else:  # early_stage
            return x * 10  # Early stage gets higher score

    def hit_condition(self, d: Dict) -> bool:
        """Check if this reaction node represents a convergent Suzuki coupling."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactant_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if this looks like a Suzuki coupling
            if not self._is_suzuki_coupling(product, reactants):
                return False
                
            # Check convergency - reactants should be substantial fragments
            return self._is_convergent_assembly(reactants)
            
        except Exception:
            return False

    def _is_suzuki_coupling(self, product: Chem.Mol, reactants: List[Chem.Mol]) -> bool:
        """Check if reaction pattern matches Suzuki coupling."""
        # Must have exactly 2 main organic reactants (ignoring catalysts/bases)
        organic_reactants = [r for r in reactants if r.GetNumHeavyAtoms() > 3]
        if len(organic_reactants) != 2:
            return False
            
        reactant1, reactant2 = organic_reactants
        
        # One reactant should have boronic acid/ester, other should have halide
        has_boronic = any(reactant1.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                         for p in self.boronic_patterns)
        has_halide = any(reactant2.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                        for p in self.halide_patterns)
        
        if not (has_boronic and has_halide):
            # Try the other way around
            has_boronic = any(reactant2.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                             for p in self.boronic_patterns)
            has_halide = any(reactant1.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                            for p in self.halide_patterns)
                            
        if not (has_boronic and has_halide):
            return False
            
        # Product should show C-C coupling pattern
        return any(product.HasSubstructMatch(Chem.MolFromSmarts(p)) 
                  for p in self.suzuki_patterns)

    def _is_convergent_assembly(self, reactants: List[Chem.Mol]) -> bool:
        """Check if reactants represent convergent fragments (not linear extension)."""
        organic_reactants = [r for r in reactants if r.GetNumHeavyAtoms() > 3]
        
        if len(organic_reactants) != 2:
            return False
            
        # Both fragments should be substantial (>= 6 heavy atoms for convergency)
        min_fragment_size = 6
        fragment_sizes = [r.GetNumHeavyAtoms() for r in organic_reactants]
        
        # Check size requirements
        if not all(size >= min_fragment_size for size in fragment_sizes):
            return False
            
        # Check size balance - fragments shouldn't be too different in size
        # (avoid cases where one is just a small coupling partner)
        size_ratio = min(fragment_sizes) / max(fragment_sizes)
        return size_ratio >= 0.3  # At least 30% size ratio for convergency
