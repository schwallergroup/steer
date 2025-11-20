"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if two major fragments
    are coupled via a specific reaction type at a target depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_reaction = config["coupling_reaction"]
        self.convergence_step = config["convergence_step"]
        
        # Define SMARTS patterns for different coupling reactions
        self.coupling_patterns = {
            "amide_formation": "[C:1](=[O:2])[NH:3]",
            "suzuki_coupling": "[c:1][c:2]",
            "click_chemistry": "[c:1]1[nH:2][nH:3][nH:4][c:5]1",
            "ester_formation": "[C:1](=[O:2])[O:3][C:4]",
            "c_n_coupling": "[c:1][NH:2]"
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        # Score based on how close the convergence is to target step
        target_fraction = self.convergence_step / 10.0  # Normalize target step
        
        if abs(x - target_fraction) < 0.1:  # Within 1 step of target
            return 10
        elif abs(x - target_fraction) < 0.2:  # Within 2 steps
            return 8
        elif abs(x - target_fraction) < 0.3:  # Within 3 steps
            return 6
        else:
            return max(0, 4 - abs(x - target_fraction) * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling step"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            # Check if we have the expected number of reactant fragments
            reactants = reactants_smiles.split(".")
            if len(reactants) != self.fragment_count:
                return False
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product or not all(reactant_mols):
                return False
            
            # Check if the coupling pattern is present in the product
            coupling_pattern = self.coupling_patterns.get(self.coupling_reaction)
            if not coupling_pattern:
                return False
                
            pattern_mol = Chem.MolFromSmarts(coupling_pattern)
            if not pattern_mol:
                return False
                
            # Verify the coupling pattern exists in product but not in individual reactants
            product_has_pattern = product.HasSubstructMatch(pattern_mol)
            reactants_have_pattern = any(mol.HasSubstructMatch(pattern_mol) for mol in reactant_mols)
            
            # Check fragment complexity (each reactant should have reasonable size)
            min_fragment_size = 5  # Minimum atoms for a "major fragment"
            fragments_large_enough = all(mol.GetNumAtoms() >= min_fragment_size for mol in reactant_mols)
            
            # True convergent coupling: pattern formed in product, fragments are substantial
            return (product_has_pattern and not reactants_have_pattern and 
                   fragments_large_enough and self._check_fragment_balance(reactant_mols))
            
        except Exception:
            return False
    
    def _check_fragment_balance(self, reactant_mols) -> bool:
        """Check that fragments are reasonably balanced in size (not one tiny, one huge)"""
        sizes = [mol.GetNumAtoms() for mol in reactant_mols]
        if len(sizes) < 2:
            return False
            
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Ratio shouldn't be more than 4:1 for balanced convergent synthesis
        return (max_size / min_size) <= 4.0
