"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting coupling reactions
    that join major fragments. Rewards early-stage convergent steps.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_reactions = config["coupling_reactions"]
        self.early_stage = config["early_stage"]
        
        # Define SMARTS patterns for coupling reactions
        self.reaction_patterns = {
            "suzuki": ["[#6]-[#5]", "[#6]-[#6].[#5]", "[c,C]-c1ccccc1"],  # Suzuki coupling patterns
            "heck": ["[C]=[C]", "[#6]-[#6]=[#6]"],  # Heck coupling patterns
            "sonogashira": ["[C]#[C]", "[#6]-[C]#[C]"],  # Sonogashira patterns
            "buchwald": ["[c,C]-[N]", "[#6]-[#7]"],  # Buchwald-Hartwig patterns
        }

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.early_stage:
            # Reward early-stage convergent steps (lower depth is better)
            return max(0, 10 * (1 - x))
        else:
            # Neutral scoring - just presence matters
            return 5

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a convergent coupling reaction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Split reactants
            reactants = reactants_smiles.split(".")
            
            # Check if we have the expected number of major fragments
            major_fragments = self._identify_major_fragments(reactants)
            
            if len(major_fragments) < self.fragment_count:
                return False
                
            # Check if any of the specified coupling reactions occurred
            return self._detect_coupling_reaction(product_smiles, major_fragments)
            
        except Exception:
            return False

    def _identify_major_fragments(self, reactants: List[str]) -> List[str]:
        """
        Identify major fragments by molecular weight and complexity.
        Filters out small molecules like catalysts, bases, solvents.
        """
        major_fragments = []
        min_atoms = 8  # Minimum number of heavy atoms for a major fragment
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                
                # Skip small molecules (catalysts, bases, etc.)
                if num_heavy_atoms < min_atoms:
                    continue
                    
                # Skip common reagents/catalysts
                if self._is_common_reagent(mol):
                    continue
                    
                major_fragments.append(reactant_smiles)
                
            except Exception:
                continue
                
        return major_fragments

    def _is_common_reagent(self, mol) -> bool:
        """
        Check if molecule is a common reagent/catalyst that shouldn't count as a major fragment.
        """
        common_reagent_patterns = [
            "[Pd]",  # Palladium catalysts
            "[B]([OH])[OH]",  # Boronic acids (small ones)
            "[Li]", "[Na]", "[K]",  # Alkali metals
            "C(=O)[O-]",  # Carbonates/acetates
            "[N+](C)(C)(C)C.[F-]",  # TBAF
        ]
        
        for pattern in common_reagent_patterns:
            try:
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                    return True
            except Exception:
                continue
                
        return False

    def _detect_coupling_reaction(self, product_smiles: str, reactant_fragments: List[str]) -> bool:
        """
        Detect if the specified coupling reactions occurred by checking for
        characteristic bond formations.
        """
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            # Check each specified coupling reaction type
            for coupling_type in self.coupling_reactions:
                if coupling_type in self.reaction_patterns:
                    patterns = self.reaction_patterns[coupling_type]
                    
                    for pattern in patterns:
                        try:
                            pattern_mol = Chem.MolFromSmarts(pattern)
                            if pattern_mol and product_mol.HasSubstructMatch(pattern_mol):
                                # Additional check: ensure this bond wasn't present in individual reactants
                                if self._is_new_bond_formation(pattern, reactant_fragments):
                                    return True
                        except Exception:
                            continue
                            
        except Exception:
            pass
            
        return False

    def _is_new_bond_formation(self, pattern: str, reactant_fragments: List[str]) -> bool:
        """
        Check if the detected pattern represents a new bond formation
        (not present in individual reactants).
        """
        try:
            pattern_mol = Chem.MolFromSmarts(pattern)
            if pattern_mol is None:
                return True  # Default to true if pattern parsing fails
                
            # Check if this pattern exists in any single reactant
            for fragment_smiles in reactant_fragments:
                fragment_mol = Chem.MolFromSmiles(fragment_smiles)
                if fragment_mol and fragment_mol.HasSubstructMatch(pattern_mol):
                    return False  # Pattern already existed in a reactant
                    
            return True  # Pattern is new, indicating bond formation
            
        except Exception:
            return True  # Default to true on error
