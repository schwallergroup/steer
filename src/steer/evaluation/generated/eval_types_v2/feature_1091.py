"""Generated evaluation code for: Convergent synthesis via two main fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates if a synthesis route follows a convergent strategy by identifying
    a coupling reaction (like Negishi) that joins two complex fragments.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "Negishi")
        self.complexity_threshold = config.get("complexity_threshold", 5)
        
        # Define reaction patterns for different coupling reactions
        self.coupling_patterns = {
            "Negishi": ["[#6]-[Zn]", "[#6X3]([Cl,Br,I])"],
            "Suzuki": ["[#6]-[B]", "[#6X3]([Cl,Br,I])"],
            "Stille": ["[#6]-[Sn]", "[#6X3]([Cl,Br,I])"],
            "Heck": ["[#6]=[#6]", "[#6X3]([Cl,Br,I])"]
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Earlier convergent coupling is better (more convergent)
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling step."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or len(reactants) < self.fragment_count:
                return False
            
            # Check if it's the specified coupling reaction type
            if not self._is_coupling_reaction(reactants):
                return False
            
            # Check if we have the required number of complex fragments
            complex_fragments = self._count_complex_fragments(reactants)
            
            return complex_fragments >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, reactants) -> bool:
        """Check if reactants match the coupling reaction pattern."""
        if self.coupling_reaction not in self.coupling_patterns:
            return True  # If reaction type not defined, assume it's valid
            
        patterns = self.coupling_patterns[self.coupling_reaction]
        pattern_matches = [False] * len(patterns)
        
        for reactant in reactants:
            if not reactant:
                continue
                
            for i, pattern in enumerate(patterns):
                pattern_mol = Chem.MolFromSmarts(pattern)
                if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                    pattern_matches[i] = True
        
        # For coupling reactions, we need at least one match from each pattern type
        return all(pattern_matches)
    
    def _count_complex_fragments(self, reactants) -> int:
        """Count reactants that exceed the complexity threshold."""
        complex_count = 0
        
        for reactant in reactants:
            if not reactant:
                continue
                
            # Skip small molecules (catalysts, bases, etc.)
            if reactant.GetNumAtoms() < 5:
                continue
                
            complexity = self._calculate_complexity(reactant)
            if complexity >= self.complexity_threshold:
                complex_count += 1
                
        return complex_count
    
    def _calculate_complexity(self, mol) -> float:
        """Calculate molecular complexity based on rings, stereocenters, and size."""
        if not mol:
            return 0
            
        # Basic complexity metrics
        num_atoms = mol.GetNumAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        num_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        
        # Simple complexity score
        complexity = num_atoms * 0.1 + num_rings * 2 + num_stereo * 1.5
        
        return complexity
