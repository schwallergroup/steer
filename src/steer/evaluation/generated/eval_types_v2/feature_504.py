"""Generated evaluation code for: Convergent synthesis via two fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates whether a synthesis route follows a convergent strategy by coupling
    two fragments at a specified position in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_position = config.get("coupling_step_position", "final")
        self.min_fragment_complexity = config.get("min_fragment_complexity", 5)  # atoms
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No convergent coupling found
        
        if self.coupling_position == "final":
            # Reward coupling at the very end (depth close to 0)
            return 10 * (1 - x)
        elif self.coupling_position == "late":
            # Reward coupling in the last 30% of the route
            if x <= 0.3:
                return 10 * (0.3 - x) / 0.3
            else:
                return 0
        else:  # "any"
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a convergent coupling step"""
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
        if len(reactant_smiles_list) < self.fragment_count:
            return False
            
        # Filter out small molecules (catalysts, reagents)
        significant_reactants = []
        for r_smiles in reactant_smiles_list:
            mol = Chem.MolFromSmiles(r_smiles)
            if mol and mol.GetNumAtoms() >= self.min_fragment_complexity:
                significant_reactants.append(mol)
        
        # Check if we have at least the required number of significant fragments
        if len(significant_reactants) < self.fragment_count:
            return False
            
        # Verify that the fragments are actually being coupled
        # (not just mixed - they should share atoms in the product)
        product_mol = Chem.MolFromSmiles(product_smiles)
        if not product_mol:
            return False
            
        # Check atom mapping to confirm fragments are joined
        product_maps = set(atom.GetAtomMapNum() for atom in product_mol.GetAtoms() 
                          if atom.GetAtomMapNum() > 0)
        
        fragment_maps = []
        for reactant in significant_reactants:
            maps = set(atom.GetAtomMapNum() for atom in reactant.GetAtoms() 
                      if atom.GetAtomMapNum() > 0)
            if maps:  # Only consider mapped fragments
                fragment_maps.append(maps)
        
        # Ensure we have at least 2 mapped fragments and all maps appear in product
        if len(fragment_maps) >= self.fragment_count:
            all_reactant_maps = set()
            for maps in fragment_maps:
                all_reactant_maps.update(maps)
            
            # Most reactant atoms should appear in product (allowing for some loss)
            overlap = len(all_reactant_maps.intersection(product_maps))
            return overlap >= 0.8 * len(all_reactant_maps)
            
        return False
