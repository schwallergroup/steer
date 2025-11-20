"""Generated evaluation code for: Convergent synthesis via three fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentThreeFragmentAssembly(BaseScoring):
    """
    Evaluates convergent synthesis strategy where three major fragments are assembled
    in late-stage coupling reactions. Checks for the presence of fragment assembly
    reactions and penalizes early assembly.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.assembly_stage = config["assembly_stage"]
        self.min_fragment_size = config.get("min_fragment_size", 6)  # atoms
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent assembly found
        
        if self.assembly_stage == "late":
            # Late-stage assembly is preferred (closer to 1.0)
            return 1.0 - x
        else:
            # Early-stage assembly penalty
            return x if x < 0.5 else 0.5 - (x - 0.5)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent assembly of multiple fragments
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            # Parse reaction
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Filter reactants by size (exclude small molecules like reagents)
            significant_reactants = []
            for reactant in reactants:
                if reactant.GetNumAtoms() >= self.min_fragment_size:
                    significant_reactants.append(reactant)
            
            # Check if we have the required number of fragments
            if len(significant_reactants) != self.fragment_count:
                return False
                
            # Verify this is actually a coupling reaction (fragments are combined)
            return self._is_fragment_coupling(product, significant_reactants)
            
        except Exception:
            return False
    
    def _is_fragment_coupling(self, product, reactants) -> bool:
        """
        Verify that reactants are being coupled together to form the product
        by checking that significant portions of each reactant appear in the product
        """
        if len(reactants) < 2:
            return False
            
        # Check that each reactant contributes a significant portion to the product
        for reactant in reactants:
            if not self._fragment_preserved_in_product(reactant, product):
                return False
                
        return True
    
    def _fragment_preserved_in_product(self, fragment, product) -> bool:
        """
        Check if a significant portion of the fragment structure is preserved in the product
        """
        # Create a substructure pattern from the fragment (remove 1-2 atoms for flexibility)
        fragment_size = fragment.GetNumAtoms()
        min_match_size = max(4, fragment_size - 2)
        
        # Simple approach: check if most of the fragment atoms have matching environments
        # This is a simplified version - in practice, you might want more sophisticated matching
        
        # Get all possible substructures of the fragment
        fragment_fp = Chem.RDKFingerprint(fragment)
        product_fp = Chem.RDKFingerprint(product)
        
        # Calculate Tanimoto similarity
        similarity = DataStructs.TanimotoSimilarity(fragment_fp, product_fp)
        
        # If similarity is reasonable, assume fragment is preserved
        return similarity > 0.3  # Threshold for fragment preservation
