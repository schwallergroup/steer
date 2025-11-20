"""Generated evaluation code for: Convergent synthesis via two complex fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two complex fragments
    are coupled together via cross-coupling reactions at an optimal depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "cross_coupling")
        self.complexity_level = config.get("complexity_level", "high")
        self.target_depth = config.get("target_depth", {"type": "range", "min": 0.2, "max": 0.6})
        
        # Cross-coupling reaction SMARTS patterns
        self.coupling_patterns = {
            "cross_coupling": [
                "[#6:1]-[Br,I,Cl].[#6:2]-[B]>>([#6:1]-[#6:2])",  # Suzuki
                "[#6:1]-[Br,I].[#6:2]-[Sn]>>([#6:1]-[#6:2])",    # Stille
                "[#6:1]-[Br,I].[#6:2]=[CH2]>>([#6:1]-[#6:2])",   # Heck
                "[#6:1]-[Br,I].[#6:2]#[CH]>>([#6:1]-[#6:2])",    # Sonogashira
            ]
        }
        
        # Complexity indicators for fragments
        self.complexity_indicators = [
            "[r5]",  # 5-membered rings
            "[r6]",  # 6-membered rings
            "[nH]",  # NH in rings (indazoles, etc.)
            "[#7]=[#6]",  # C=N bonds
            "[#6]1:[#6]:[#6]:[#7]:[#6]:[#6]:1",  # pyridine-like
            "[#7]1:[#7]:[#6]:[#6]:[#6]:1",  # indazole-like
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't occur
        
        if self.target_depth["type"] == "range":
            min_depth = self.target_depth["min"]
            max_depth = self.target_depth["max"]
            
            if min_depth <= x <= max_depth:
                return 10  # Optimal convergent depth
            elif x < min_depth:
                return 8 * (x / min_depth)  # Too early, penalize
            else:
                return 8 * max(0, (1 - x) / (1 - max_depth))  # Too late, penalize
        else:
            # Simple distance-based scoring
            target = self.target_depth.get("value", 0.4)
            return max(0, 10 - 20 * abs(x - target))

    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling of complex fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if len(reactants) < self.fragment_count:
                return False
            
            # Check if this is a cross-coupling reaction
            if not self._is_cross_coupling_reaction(mapped_rxn):
                return False
            
            # Check if we have the required number of complex fragments
            complex_fragments = []
            for reactant in reactants:
                if self._is_complex_fragment(reactant):
                    complex_fragments.append(reactant)
            
            return len(complex_fragments) >= self.fragment_count
            
        except Exception:
            return False

    def _is_cross_coupling_reaction(self, mapped_rxn: str) -> bool:
        """Check if the reaction matches cross-coupling patterns"""
        try:
            rxn_patterns = self.coupling_patterns.get(self.coupling_reaction, [])
            
            for pattern in rxn_patterns:
                try:
                    rxn_template = AllChem.ReactionFromSmarts(pattern)
                    if rxn_template:
                        # Simple heuristic: check for presence of typical cross-coupling elements
                        reactant_smiles = mapped_rxn.split(">>")[1]
                        
                        # Check for halides and organometallic coupling partners
                        has_halide = any(x in reactant_smiles for x in ['Br', 'I', 'Cl'])
                        has_metal = any(x in reactant_smiles for x in ['B', 'Sn', 'Zn'])
                        
                        if has_halide and (has_metal or self._has_alkene_alkyne(reactant_smiles)):
                            return True
                except:
                    continue
                    
            return False
        except Exception:
            return False

    def _has_alkene_alkyne(self, smiles: str) -> bool:
        """Check for alkenes or alkynes (Heck, Sonogashira reactions)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Look for C=C or C≡C bonds
            alkene_pattern = Chem.MolFromSmarts("C=C")
            alkyne_pattern = Chem.MolFromSmarts("C#C")
            
            return (mol.HasSubstructMatch(alkene_pattern) or 
                   mol.HasSubstructMatch(alkyne_pattern))
        except:
            return False

    def _is_complex_fragment(self, mol) -> bool:
        """Determine if a molecule qualifies as a complex fragment"""
        if mol is None:
            return False
            
        try:
            # Count complexity indicators
            complexity_score = 0
            
            # Ring systems
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            complexity_score += num_rings * 2
            
            # Heteroatoms in rings
            for pattern_smarts in self.complexity_indicators:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    complexity_score += 1
            
            # Size requirement (complex fragments should be reasonably sized)
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms < 8:  # Too small to be "complex"
                return False
            
            # Complexity thresholds based on level
            if self.complexity_level == "high":
                return complexity_score >= 4 and num_rings >= 2
            elif self.complexity_level == "medium":
                return complexity_score >= 2 and num_rings >= 1
            else:  # low
                return complexity_score >= 1
                
        except Exception:
            return False
