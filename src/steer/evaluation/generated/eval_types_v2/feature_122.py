"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Suzuki-Miyaura cross-coupling
    to join two complex fragments at a late stage in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.stage = config.get("stage", "late")
        self.target_depth_fraction = 0.2 if self.stage == "late" else 0.5
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        if self.stage == "late":
            # Reward late-stage coupling (lower depth fraction is better)
            if x <= self.target_depth_fraction:
                return 10  # Perfect late-stage coupling
            else:
                # Penalty increases as coupling moves earlier
                return max(0, 10 - 20 * (x - self.target_depth_fraction))
        else:
            # For mid-stage coupling
            return max(0, 10 - 10 * abs(x - self.target_depth_fraction))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling with appropriate convergence"""
        metadata = d.get("metadata", {})
        
        # Check if reaction contains Suzuki coupling pattern
        if not self._is_suzuki_coupling(metadata):
            return False
        
        # Check convergence: reaction should have multiple reactants of similar complexity
        return self._is_convergent_reaction(metadata)
    
    def _is_suzuki_coupling(self, metadata) -> bool:
        """Detect Suzuki-Miyaura coupling reaction"""
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants = rxn_parts[0].split(".")
            product = rxn_parts[1]
            
            # Look for boronic acid/ester patterns in reactants
            boronic_patterns = [
                "[#6]-B(-O)-O",  # Boronic acid
                "[#6]-B1-O-C-C-O-1",  # Boronic ester (pinacol)
                "[#6]-B(-[OH])-[OH]"  # Boronic acid explicit
            ]
            
            # Look for aryl halide patterns
            halide_patterns = [
                "[#6]:[#6]-[Br]",  # Aryl bromide
                "[#6]:[#6]-[I]",   # Aryl iodide
                "[#6]:[#6]-[Cl]"   # Aryl chloride
            ]
            
            has_boronic = False
            has_halide = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                
                # Check for boronic acid/ester
                for pattern in boronic_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_boronic = True
                        break
                
                # Check for aryl halide
                for pattern in halide_patterns:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_halide = True
                        break
            
            return has_boronic and has_halide
            
        except Exception:
            return False
    
    def _is_convergent_reaction(self, metadata) -> bool:
        """Check if reaction shows convergent behavior (similar fragment complexity)"""
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
        
        try:
            reactants = mapped_rxn.split(">>")[0].split(".")
            
            if len(reactants) < self.fragment_count:
                return False
            
            # Calculate complexity (heavy atom count) for each reactant
            complexities = []
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    # Skip small molecules (catalysts, bases, etc.)
                    heavy_atoms = mol.GetNumHeavyAtoms()
                    if heavy_atoms >= 5:  # Minimum fragment size
                        complexities.append(heavy_atoms)
            
            if len(complexities) < self.fragment_count:
                return False
            
            # Sort by complexity and check top fragments
            complexities.sort(reverse=True)
            top_fragments = complexities[:self.fragment_count]
            
            # Check if fragments are reasonably similar in complexity (convergent)
            if len(top_fragments) >= 2:
                ratio = min(top_fragments) / max(top_fragments)
                return ratio >= 0.3  # At least 30% complexity ratio
            
            return False
            
        except Exception:
            return False
