"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if major fragments
    are coupled at a specific depth in the synthesis route.
    
    Looks for reactions where multiple complex fragments (non-trivial starting materials)
    are joined together, indicating a convergent approach rather than linear synthesis.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.target_coupling_depth = config.get("coupling_step_depth", 2)
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)  # min atoms for "complex" fragment
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Better score if coupling happens at target depth
            depth_penalty = abs(x - self.target_coupling_depth / 10.0)
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        A convergent step joins multiple complex fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Parse reactants
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = []
            
            for smi in reactant_smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactants.append(mol)
            
            # Check if we have the required number of complex fragments
            complex_fragments = []
            for mol in reactants:
                if self._is_complex_fragment(mol):
                    complex_fragments.append(mol)
            
            # Must have at least the target fragment count
            if len(complex_fragments) < self.fragment_count:
                return False
                
            # Verify this is actually a coupling reaction (fragments are joined)
            return self._is_coupling_reaction(complex_fragments, product_smiles)
            
        except Exception:
            return False
    
    def _is_complex_fragment(self, mol) -> bool:
        """
        Determine if a molecule is a complex fragment worthy of convergent synthesis.
        Uses atom count and structural complexity as heuristics.
        """
        if mol is None:
            return False
            
        atom_count = mol.GetNumAtoms()
        if atom_count < self.min_fragment_complexity:
            return False
            
        # Additional complexity checks
        ring_count = mol.GetRingInfo().NumRings()
        bond_count = mol.GetNumBonds()
        
        # Simple heuristic: complex if it has rings or many atoms
        complexity_score = atom_count + (ring_count * 3)
        
        return complexity_score >= self.min_fragment_complexity
    
    def _is_coupling_reaction(self, fragments, product_smiles) -> bool:
        """
        Verify that the fragments are actually being coupled together
        by checking if key atoms from each fragment appear in the product.
        """
        try:
            product = Chem.MolFromSmiles(product_smiles)
            if product is None:
                return False
                
            # Get atom map numbers from fragments
            fragment_maps = []
            for frag in fragments:
                frag_atoms = [atom.GetAtomMapNum() for atom in frag.GetAtoms() 
                            if atom.GetAtomMapNum() > 0]
                if frag_atoms:
                    fragment_maps.append(set(frag_atoms))
            
            if len(fragment_maps) < self.fragment_count:
                return False
                
            # Check if atoms from different fragments appear together in product
            product_atoms = set(atom.GetAtomMapNum() for atom in product.GetAtoms() 
                              if atom.GetAtomMapNum() > 0)
            
            # Verify atoms from each fragment are present in product
            fragments_represented = 0
            for frag_atoms in fragment_maps:
                if frag_atoms.intersection(product_atoms):
                    fragments_represented += 1
                    
            return fragments_represented >= self.fragment_count
            
        except Exception:
            return False
