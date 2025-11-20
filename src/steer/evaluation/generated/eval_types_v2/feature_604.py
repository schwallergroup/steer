"""Generated evaluation code for: Convergent synthesis via heterocycle fragment"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentHeterocycleFragment(BaseScoring):
    """
    Evaluates convergent synthesis strategy where major fragments including 
    heterocycles (macrocycle, thienopyrimidine) are assembled together.
    
    Checks for reactions that combine pre-formed heterocyclic fragments
    rather than building them sequentially.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.major_fragments = config.get("major_fragments", [])
        
        # Define SMARTS patterns for target heterocycles
        self.fragment_patterns = {
            "macrocycle": "[R{12-}]",  # Ring size 12 or larger
            "thienopyrimidine": "c1sc2ncncc2c1",  # Thieno[2,3-d]pyrimidine core
            "pyrimidine": "c1ncncn1",
            "thiophene": "c1sccc1",
            "pyridine": "c1ccncc1",
            "imidazole": "c1[nH]cnc1",
            "benzimidazole": "c1ccc2[nH]cnc2c1"
        }
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth of convergent fragment assembly.
        Earlier convergent steps (higher depth fraction) score better.
        """
        if x < 0:
            return 0  # No convergent heterocycle assembly found
        else:
            # Earlier convergent assembly is better (closer to 1.0 depth fraction)
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents convergent assembly of heterocyclic fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".") if r.strip()]
            
            if not product or len(reactants) < self.fragment_count:
                return False
            
            # Check if product contains target heterocycles
            product_has_targets = self._contains_target_fragments(product)
            if not product_has_targets:
                return False
            
            # Check if reactants represent pre-formed heterocyclic fragments
            fragment_reactants = 0
            target_fragments_found = set()
            
            for reactant in reactants:
                if self._is_heterocycle_fragment(reactant):
                    fragment_reactants += 1
                    
                    # Check which specific target fragments this reactant contains
                    for fragment_name in self.major_fragments:
                        if self._contains_specific_fragment(reactant, fragment_name):
                            target_fragments_found.add(fragment_name)
            
            # Must have at least the required fragment count and contain major fragments
            is_convergent = fragment_reactants >= self.fragment_count
            has_major_fragments = len(target_fragments_found) >= len(self.major_fragments) * 0.5
            
            return is_convergent and has_major_fragments
            
        except Exception:
            return False
    
    def _contains_target_fragments(self, mol) -> bool:
        """Check if molecule contains any of the target heterocyclic fragments."""
        if not mol:
            return False
            
        for fragment_name in self.major_fragments:
            if fragment_name in self.fragment_patterns:
                pattern = Chem.MolFromSmarts(self.fragment_patterns[fragment_name])
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
        return False
    
    def _contains_specific_fragment(self, mol, fragment_name) -> bool:
        """Check if molecule contains a specific heterocyclic fragment."""
        if not mol or fragment_name not in self.fragment_patterns:
            return False
            
        pattern = Chem.MolFromSmarts(self.fragment_patterns[fragment_name])
        return pattern and mol.HasSubstructMatch(pattern)
    
    def _is_heterocycle_fragment(self, mol) -> bool:
        """
        Check if molecule is a substantial heterocyclic fragment.
        Must contain at least one heterocycle and meet size requirements.
        """
        if not mol:
            return False
            
        # Must have at least one heteroatom in a ring
        has_hetero_ring = False
        for ring in mol.GetRingInfo().AtomRings():
            for atom_idx in ring:
                atom = mol.GetAtomWithIdx(atom_idx)
                if atom.GetSymbol() in ['N', 'O', 'S']:
                    has_hetero_ring = True
                    break
            if has_hetero_ring:
                break
        
        if not has_hetero_ring:
            return False
            
        # Must be substantial (at least 6 heavy atoms)
        heavy_atom_count = mol.GetNumHeavyAtoms()
        
        return heavy_atom_count >= 6
