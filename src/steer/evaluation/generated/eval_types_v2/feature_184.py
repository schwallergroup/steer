"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two fragments are synthesized 
    separately and coupled together in a late-stage reaction.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.late_stage_coupling = config["late_stage_coupling"]
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)  # minimum atom count
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        else:
            if self.late_stage_coupling:
                return 1 - x  # Earlier coupling is better for late-stage strategy
            else:
                return 1  # Any convergent coupling is good
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling of two fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        # Parse reactants
        reactant_smiles_list = reactants_smiles.split(".")
        if len(reactant_smiles_list) < 2:
            return False  # Need at least 2 reactants for convergent coupling
            
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles_list]
            
            if not product or not all(reactants):
                return False
                
            # Filter reactants by complexity (exclude small reagents/catalysts)
            complex_reactants = []
            for reactant in reactants:
                if reactant.GetNumHeavyAtoms() >= self.min_fragment_complexity:
                    complex_reactants.append(reactant)
                    
            # Check if we have exactly the target number of complex fragments
            if len(complex_reactants) != self.fragment_count:
                return False
                
            # Verify this is a true coupling reaction (fragments combine to form product)
            return self._is_coupling_reaction(product, complex_reactants)
            
        except Exception:
            return False
    
    def _is_coupling_reaction(self, product, reactants) -> bool:
        """
        Verify that the reactants are actually coupled together to form the product.
        This checks that the combined atom count of reactants approximately equals product.
        """
        product_heavy_atoms = product.GetNumHeavyAtoms()
        reactants_heavy_atoms = sum(r.GetNumHeavyAtoms() for r in reactants)
        
        # Allow for small differences due to leaving groups or added atoms
        atom_difference = abs(product_heavy_atoms - reactants_heavy_atoms)
        
        # The difference should be small (accounting for typical coupling reactions)
        return atom_difference <= 3
