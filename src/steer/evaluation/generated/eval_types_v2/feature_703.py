"""Generated evaluation code for: Intramolecular nitrene cyclization for ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class IntramolecularNitreneCyclization(BaseScoring):
    """
    Evaluates synthesis routes for the presence of intramolecular nitrene cyclization reactions.
    
    This class detects azide-to-nitrene cyclization reactions that form rings through
    intramolecular mechanisms, typically used to create tricyclic core structures.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "mid")
        self.target_depth = 0.5  # Mid-stage timing
        if self.timing == "early":
            self.target_depth = 0.8
        elif self.timing == "late":
            self.target_depth = 0.2
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Score based on timing preference
            timing_score = 1 - abs(x - self.target_depth)
            return max(0, timing_score * 10)
    
    def hit_condition(self, d):
        """
        Detects intramolecular nitrene cyclization reactions.
        
        Looks for:
        1. Azide group ([N-][N+]#N) in reactant
        2. Formation of new N-containing ring in product
        3. Intramolecular mechanism (same molecule cyclizes)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define azide pattern
            azide_pattern = Chem.MolFromSmarts("[N-][N+]#N")
            if not azide_pattern:
                return False
            
            # Check for azide in reactants
            has_azide_reactant = False
            azide_containing_reactant = None
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(azide_pattern):
                    has_azide_reactant = True
                    azide_containing_reactant = reactant
                    break
            
            if not has_azide_reactant or not azide_containing_reactant:
                return False
            
            # Count rings in reactant vs product for the molecule that contained azide
            reactant_ring_count = Chem.rdMolDescriptors.CalcNumRings(azide_containing_reactant)
            
            # Find corresponding product by matching atom map numbers
            reactant_atoms = set(atom.GetAtomMapNum() for atom in azide_containing_reactant.GetAtoms() 
                               if atom.GetAtomMapNum() > 0)
            
            corresponding_product = None
            for product in products:
                product_atoms = set(atom.GetAtomMapNum() for atom in product.GetAtoms() 
                                  if atom.GetAtomMapNum() > 0)
                if reactant_atoms.intersection(product_atoms):
                    corresponding_product = product
                    break
            
            if not corresponding_product:
                return False
            
            product_ring_count = Chem.rdMolDescriptors.CalcNumRings(corresponding_product)
            
            # Check for ring formation (increase in ring count)
            ring_formed = product_ring_count > reactant_ring_count
            
            # Check for nitrogen incorporation in new ring
            nitrogen_in_rings = False
            if ring_formed:
                # Look for nitrogen atoms in ring systems
                ring_info = corresponding_product.GetRingInfo()
                for ring in ring_info.AtomRings():
                    for atom_idx in ring:
                        if corresponding_product.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7:
                            nitrogen_in_rings = True
                            break
                    if nitrogen_in_rings:
                        break
            
            # Verify azide is consumed (no longer present in product)
            azide_consumed = not corresponding_product.HasSubstructMatch(azide_pattern)
            
            return has_azide_reactant and ring_formed and nitrogen_in_rings and azide_consumed
            
        except Exception:
            return False
