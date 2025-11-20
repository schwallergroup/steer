"""Generated evaluation code for: Convergent synthesis via Suzuki coupling fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzuki(BaseScoring):
    """
    Evaluates convergent synthesis strategies using Suzuki coupling as the final step
    to join two complex pre-assembled fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.timing = config["parameters"].get("timing", "final")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            if self.timing == "final":
                # Reward early occurrence (lower depth) for final timing
                return 1 - x
            else:
                # For other timing preferences, could implement different scoring
                return 1 - x
                
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling between appropriate fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Check if this is a Suzuki coupling reaction
        if not self._is_suzuki_coupling(mapped_rxn):
            return False
            
        # Check if it's coupling the expected number of fragments
        return self._has_correct_fragment_count(mapped_rxn)
        
    def _is_suzuki_coupling(self, mapped_rxn: str) -> bool:
        """Detect Suzuki coupling by looking for boronic acid/ester + aryl halide pattern"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[1].split(".")
            
            # Patterns for Suzuki coupling components
            boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
            boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)C(C)(C)O1")  # Pinacol ester
            aryl_halide_pattern = Chem.MolFromSmarts("c-[Cl,Br,I]")
            triflate_pattern = Chem.MolFromSmarts("c-OS(=O)(=O)C(F)(F)F")
            
            has_boron_component = False
            has_electrophile = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                # Check for boronic acid/ester
                if (mol.HasSubstructMatch(boronic_acid_pattern) or 
                    mol.HasSubstructMatch(boronic_ester_pattern)):
                    has_boron_component = True
                    
                # Check for aryl halide or triflate
                if (mol.HasSubstructMatch(aryl_halide_pattern) or 
                    mol.HasSubstructMatch(triflate_pattern)):
                    has_electrophile = True
                    
            return has_boron_component and has_electrophile
            
        except Exception:
            return False
            
    def _has_correct_fragment_count(self, mapped_rxn: str) -> bool:
        """Check if the reaction couples the expected number of fragments"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants = rxn_parts[1].split(".")
            
            # Filter out small molecules (catalysts, bases, solvents)
            complex_fragments = []
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                # Consider a fragment "complex" if it has sufficient size/complexity
                atom_count = mol.GetNumAtoms()
                ring_count = Chem.rdMolDescriptors.CalcNumRings(mol)
                
                # Skip small molecules likely to be reagents
                if atom_count < 6:
                    continue
                    
                # Skip common reagents/catalysts
                if self._is_likely_reagent(mol):
                    continue
                    
                complex_fragments.append(reactant_smiles)
                
            return len(complex_fragments) >= self.fragment_count
            
        except Exception:
            return False
            
    def _is_likely_reagent(self, mol) -> bool:
        """Identify common Suzuki coupling reagents to exclude from fragment count"""
        # Common bases and catalysts
        reagent_patterns = [
            "[Na+]",  # Sodium salts
            "[K+]",   # Potassium salts  
            "N(CC)CC", # Triethylamine
            "c1ccc(P(c2ccccc2)c2ccccc2)cc1", # Triphenylphosphine derivatives
            "[Pd]",   # Palladium catalysts
        ]
        
        for pattern_smarts in reagent_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
            except:
                continue
                
        return False
