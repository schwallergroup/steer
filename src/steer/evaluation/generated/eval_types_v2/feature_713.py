"""Generated evaluation code for: Convergent synthesis via two main fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by detecting when two substantial 
    fragments are coupled together to form the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_stage = config.get("coupling_stage", "late")
        self.min_fragment_size = config.get("min_fragment_size", 8)  # minimum atoms per fragment
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_stage == "late":
            # Reward later stage convergent coupling (closer to final product)
            return 1 - x
        elif self.coupling_stage == "early":
            # Reward earlier stage convergent coupling
            return x
        else:
            # Any convergent coupling is good
            return 1
    
    def hit_condition(self, d) -> bool:
        """
        Detect if this reaction represents a convergent coupling of substantial fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactant_smiles_list = reactants_smiles.split(".")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list if smi]
            
            if not product or not reactants:
                return False
                
            # Filter reactants to find substantial fragments (not small reagents)
            substantial_fragments = []
            for reactant in reactants:
                if reactant and reactant.GetNumAtoms() >= self.min_fragment_size:
                    # Check if it's likely an organic fragment (contains carbon)
                    has_carbon = any(atom.GetSymbol() == 'C' for atom in reactant.GetAtoms())
                    if has_carbon:
                        substantial_fragments.append(reactant)
            
            # Check if we have the expected number of substantial fragments
            if len(substantial_fragments) != self.fragment_count:
                return False
                
            # Verify that fragments are being coupled (not just functionalized)
            # Check that atoms from different fragments end up bonded in product
            return self._verify_fragment_coupling(product, substantial_fragments)
            
        except Exception:
            return False
    
    def _verify_fragment_coupling(self, product, fragments) -> bool:
        """
        Verify that atoms from different fragments are actually coupled together.
        """
        try:
            # Get atom mappings for each fragment
            fragment_atom_maps = []
            for frag in fragments:
                atom_maps = set()
                for atom in frag.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        atom_maps.add(atom.GetAtomMapNum())
                if atom_maps:
                    fragment_atom_maps.append(atom_maps)
            
            if len(fragment_atom_maps) < 2:
                return False
                
            # Check for bonds between atoms from different fragments in the product
            for bond in product.GetBonds():
                begin_map = bond.GetBeginAtom().GetAtomMapNum()
                end_map = bond.GetEndAtom().GetAtomMapNum()
                
                if begin_map > 0 and end_map > 0:
                    # Check if this bond connects atoms from different fragments
                    begin_fragment = None
                    end_fragment = None
                    
                    for i, frag_maps in enumerate(fragment_atom_maps):
                        if begin_map in frag_maps:
                            begin_fragment = i
                        if end_map in frag_maps:
                            end_fragment = i
                    
                    # If atoms are from different fragments, we found a coupling bond
                    if (begin_fragment is not None and end_fragment is not None and 
                        begin_fragment != end_fragment):
                        return True
                        
            return False
            
        except Exception:
            return False
