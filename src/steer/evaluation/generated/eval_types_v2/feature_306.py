"""Generated evaluation code for: Convergent synthesis via two main fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzuki(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two main fragments are built 
    separately and joined via Suzuki coupling in the final step.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "Suzuki")
        self.coupling_step_position = config.get("coupling_step_position", "final")
        
        # Suzuki coupling SMARTS patterns
        self.suzuki_patterns = [
            "[c:1]-[B:2]([OH:3])[OH:4].[c:5]-[Br:6]>>[c:1]-[c:5]",  # Boronic acid + aryl bromide
            "[c:1]-[B:2]([OH:3])[OH:4].[c:5]-[I:6]>>[c:1]-[c:5]",   # Boronic acid + aryl iodide
            "[c:1]-[B:2]([OH:3])[OH:4].[c:5]-[Cl:6]>>[c:1]-[c:5]",  # Boronic acid + aryl chloride
            "[c:1]-[B:2]([O:3][O:4])[c:5].[c:6]-[Br:7]>>[c:1]-[c:6]", # Boronic ester + aryl bromide
        ]
        
        # Suzuki reactant patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("c-B([OH])[OH]")
        self.boronic_ester_pattern = Chem.MolFromSmarts("c-B1OCCCO1")  # Pinacol ester
        self.aryl_halide_patterns = [
            Chem.MolFromSmarts("c-Br"),
            Chem.MolFromSmarts("c-I"), 
            Chem.MolFromSmarts("c-Cl")
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        elif x == 1.0:  # Final step (perfect convergent strategy)
            return 10
        elif x > 0.8:  # Near final step
            return 8 - (1.0 - x) * 10
        else:
            return max(0, 5 - x * 5)  # Earlier steps get lower scores

    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling between two complex fragments"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            # Parse reaction
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            product_smiles = parts[0]
            reactants_smiles = parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if not product or len(reactants) < 2:
                return False
            
            # Check if this looks like a Suzuki coupling
            if not self._is_suzuki_coupling(reactants):
                return False
            
            # Check if we have the right number of main fragments (ignore small reagents)
            main_fragments = self._identify_main_fragments(reactants)
            if len(main_fragments) != self.fragment_count:
                return False
                
            # Check fragment complexity (each should have reasonable size)
            return all(self._is_complex_fragment(frag) for frag in main_fragments)
            
        except Exception:
            return False

    def _is_suzuki_coupling(self, reactants) -> bool:
        """Check if reactants match Suzuki coupling pattern"""
        has_boronic = False
        has_halide = False
        
        for reactant in reactants:
            # Check for boronic acid/ester
            if (self.boronic_acid_pattern and reactant.HasSubstructMatch(self.boronic_acid_pattern)) or \
               (self.boronic_ester_pattern and reactant.HasSubstructMatch(self.boronic_ester_pattern)):
                has_boronic = True
            
            # Check for aryl halide
            for halide_pattern in self.aryl_halide_patterns:
                if halide_pattern and reactant.HasSubstructMatch(halide_pattern):
                    has_halide = True
                    break
                    
        return has_boronic and has_halide

    def _identify_main_fragments(self, reactants) -> List:
        """Identify main synthetic fragments (exclude small reagents/catalysts)"""
        main_fragments = []
        min_heavy_atoms = 8  # Minimum size for a "main fragment"
        
        for reactant in reactants:
            num_heavy_atoms = reactant.GetNumHeavyAtoms()
            
            # Skip small molecules that are likely reagents/catalysts
            if num_heavy_atoms < min_heavy_atoms:
                continue
                
            # Skip common Suzuki reagents/catalysts by SMILES patterns
            smiles = Chem.MolToSmiles(reactant)
            if any(pattern in smiles.lower() for pattern in ['pd', 'cs2co3', 'k2co3', 'na2co3']):
                continue
                
            main_fragments.append(reactant)
            
        return main_fragments

    def _is_complex_fragment(self, fragment) -> bool:
        """Check if fragment is sufficiently complex to be a main synthetic building block"""
        # Must have reasonable size
        if fragment.GetNumHeavyAtoms() < 8:
            return False
            
        # Should have some structural complexity (rings, functional groups, etc.)
        num_rings = fragment.GetRingInfo().NumRings()
        num_heteroatoms = sum(1 for atom in fragment.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # At least some complexity required
        return num_rings >= 1 or num_heteroatoms >= 2
