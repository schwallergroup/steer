"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentStrategy(BaseScoring):
    """
    Evaluates convergent synthesis strategy by checking if the route combines
    a specified number of major fragments at a target coupling depth.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["fragment_count"]
        self.coupling_depth = config["coupling_depth"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            # Perfect score if coupling occurs at target depth
            depth_penalty = abs(x - (self.coupling_depth / 10.0))
            return max(0, 1 - depth_penalty)
    
    def hit_condition(self, d):
        """
        Check if this reaction represents a convergent coupling of major fragments.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Check if we have the expected number of fragments
            if len(reactants) != self.fragment_count:
                return False
                
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check if this is a coupling reaction (fragments combine to larger product)
            product_heavy_atoms = product_mol.GetNumHeavyAtoms()
            total_reactant_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactant_mols)
            
            # Allow for small atom count differences due to coupling chemistry
            atom_diff = abs(product_heavy_atoms - total_reactant_atoms)
            
            # Check for typical coupling patterns (amide, ester, C-C, etc.)
            is_coupling = self._detect_coupling_reaction(reactant_mols, product_mol)
            
            # Ensure fragments are substantial (not just small reagents)
            min_fragment_size = max(5, product_heavy_atoms // 4)  # At least 1/4 of product size
            substantial_fragments = all(mol.GetNumHeavyAtoms() >= min_fragment_size 
                                      for mol in reactant_mols)
            
            return (atom_diff <= 2 and is_coupling and substantial_fragments)
            
        except Exception:
            return False
    
    def _detect_coupling_reaction(self, reactants, product):
        """
        Detect common coupling reaction patterns.
        """
        # Common coupling functional groups
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[NH]")
        ester_pattern = Chem.MolFromSmarts("[C](=[O])[O][C]")
        cc_bond_pattern = Chem.MolFromSmarts("[C][C]")
        ether_pattern = Chem.MolFromSmarts("[C][O][C]")
        
        coupling_patterns = [amide_pattern, ester_pattern, cc_bond_pattern, ether_pattern]
        
        # Count coupling motifs in product vs sum in reactants
        for pattern in coupling_patterns:
            if pattern is None:
                continue
                
            product_matches = len(product.GetSubstructMatches(pattern))
            reactant_matches = sum(len(r.GetSubstructMatches(pattern)) for r in reactants)
            
            # If product has more coupling motifs than sum of reactants, likely a coupling
            if product_matches > reactant_matches:
                return True
                
        return False
