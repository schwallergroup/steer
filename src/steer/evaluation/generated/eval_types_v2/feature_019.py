"""Generated evaluation code for: Convergent assembly via ether linkage formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentEtherAssembly(BaseScoring):
    """
    Evaluates convergent assembly via ether linkage formation (Williamson ether synthesis).
    Checks if two major fragments are joined via ether formation at a specific timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"].get("fragment_count", 2)
        self.timing = config["parameters"].get("timing", "late")  # "early", "middle", "late"
        self.target_depth_fraction = self._get_target_depth_fraction()
    
    def _get_target_depth_fraction(self) -> float:
        """Convert timing preference to target depth fraction"""
        timing_map = {
            "early": 0.8,    # Early in synthesis (high depth fraction)
            "middle": 0.5,   # Middle of synthesis
            "late": 0.2      # Late in synthesis (low depth fraction)
        }
        return timing_map.get(self.timing, 0.2)
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Closer to target timing = higher score"""
        if x < 0:
            return 0  # Condition not met
        
        # Score based on how close the actual timing is to desired timing
        deviation = abs(x - self.target_depth_fraction)
        # Convert to 0-10 scale, where 0 deviation = 10, max deviation (1.0) = 0
        score = max(0, 10 * (1 - deviation))
        return score
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent ether formation"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Must have exactly the specified number of fragments
            if len(reactants) != self.fragment_count:
                return False
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if ether bond is formed
            if not self._is_ether_formation(reactant_mols, product_mol):
                return False
            
            # Check if fragments are substantial (convergent assembly)
            return self._are_substantial_fragments(reactant_mols)
            
        except Exception:
            return False
    
    def _is_ether_formation(self, reactants, product) -> bool:
        """Check if an ether bond (C-O-C) is formed in the reaction"""
        # Count ether bonds in reactants vs product
        reactant_ether_count = sum(self._count_ether_bonds(mol) for mol in reactants)
        product_ether_count = self._count_ether_bonds(product)
        
        # New ether bond should be formed
        return product_ether_count > reactant_ether_count
    
    def _count_ether_bonds(self, mol) -> int:
        """Count C-O-C ether bonds in molecule"""
        if not mol:
            return 0
        
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O':
                # Check if oxygen is bonded to exactly 2 carbons (ether)
                carbon_neighbors = [neighbor for neighbor in atom.GetNeighbors() 
                                  if neighbor.GetSymbol() == 'C']
                if len(carbon_neighbors) == 2:
                    count += 1
        return count
    
    def _are_substantial_fragments(self, reactants) -> bool:
        """Check if reactants are substantial fragments (not just small alkylating agents)"""
        min_heavy_atoms = 6  # Minimum size for a "substantial" fragment
        
        substantial_count = 0
        for mol in reactants:
            if mol.GetNumHeavyAtoms() >= min_heavy_atoms:
                substantial_count += 1
        
        # At least 2 substantial fragments for convergent assembly
        return substantial_count >= 2
