"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates whether a synthesis route follows a convergent strategy by assembling
    the target from separately prepared fragments via a coupling reaction.
    
    Checks if the route has a coupling step where two substantial fragments
    (each contributing significant complexity) are joined together.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_step = config.get("coupling_step", "final")
        self.min_fragment_atoms = config.get("min_fragment_atoms", 8)  # Minimum atoms for substantial fragment
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For convergent synthesis, earlier coupling (lower depth) is generally better
        as it indicates more parallel synthetic work.
        """
        if x < 0:
            return 0  # No convergent coupling found
        else:
            # Earlier coupling gets higher score
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling step.
        
        A convergent coupling is defined as:
        1. Multiple reactants (fragments) combining into one product
        2. Each reactant has sufficient complexity (atom count)
        3. The reaction represents a significant bond formation
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactant_smiles_list = reactants_smiles.split(".")
            
            # Need at least the specified number of fragments
            if len(reactant_smiles_list) < self.fragment_count:
                return False
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles.strip())
            reactants = []
            
            for r_smiles in reactant_smiles_list:
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol is not None:
                    reactants.append(mol)
            
            if not product or len(reactants) < self.fragment_count:
                return False
            
            # Check if we have substantial fragments (not just small reagents)
            substantial_fragments = []
            for reactant in reactants:
                # Count heavy atoms (non-hydrogen)
                heavy_atom_count = reactant.GetNumHeavyAtoms()
                if heavy_atom_count >= self.min_fragment_atoms:
                    substantial_fragments.append(reactant)
            
            # Need at least the specified number of substantial fragments
            if len(substantial_fragments) < self.fragment_count:
                return False
            
            # Additional check: ensure fragments are actually being coupled
            # (product should have more bonds than the sum of reactant bonds)
            product_bonds = product.GetNumBonds()
            reactant_bonds = sum(r.GetNumBonds() for r in reactants)
            
            # At least one new bond should be formed
            if product_bonds <= reactant_bonds:
                return False
            
            return True
            
        except Exception:
            return False
