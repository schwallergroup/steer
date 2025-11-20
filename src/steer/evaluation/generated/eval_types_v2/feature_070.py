"""Generated evaluation code for: Linear synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearSynthesisApproach(BaseScoring):
    """
    Evaluates whether a synthesis route follows a linear approach by checking
    if reactions predominantly involve single reactants rather than convergent
    fragment assembly.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.strategy_type = config["parameters"]["strategy_type"]
    
    def route_scoring(self, x) -> float:
        """
        Score based on linearity measure.
        x is the fraction of linear reactions in the route.
        Higher x means more linear approach.
        """
        if x < 0:
            return 0  # No valid route found
        
        if self.strategy_type == "linear":
            # Reward higher linearity (closer to 1.0)
            return x * 10
        else:
            # Penalize linearity if convergent approach is desired
            return (1 - x) * 10
    
    def condition_depth(self, d) -> Tuple[bool, float]:
        """
        Calculate the linearity of the entire synthesis route.
        Returns (always True to evaluate whole route, linearity_fraction)
        """
        reactions = self.get_all_reactions(d)
        
        if not reactions:
            return True, -1
        
        linear_reactions = 0
        total_reactions = len(reactions)
        
        for rxn_data in reactions:
            if self.is_linear_reaction(rxn_data):
                linear_reactions += 1
        
        linearity_fraction = linear_reactions / total_reactions
        return True, linearity_fraction
    
    def get_all_reactions(self, node) -> List:
        """Extract all reactions from the synthesis tree"""
        reactions = []
        
        def traverse(current_node):
            if "children" in current_node and current_node["children"]:
                # This node represents a reaction
                if "metadata" in current_node and "mapped_reaction_smiles" in current_node["metadata"]:
                    reactions.append(current_node)
                
                # Recursively traverse children
                for child in current_node["children"]:
                    traverse(child)
        
        traverse(node)
        return reactions
    
    def is_linear_reaction(self, rxn_data) -> bool:
        """
        Determine if a reaction is linear (involves primarily chain extension
        rather than fragment coupling).
        """
        if "metadata" not in rxn_data or "mapped_reaction_smiles" not in rxn_data["metadata"]:
            return False
        
        rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
        parts = rxn_smiles.split(">>")
        
        if len(parts) != 2:
            return False
        
        reactants_smiles = parts[1].split(".")
        product_smiles = parts[0]
        
        # Linear reactions typically involve 1-2 reactants
        # and show significant size increase from largest reactant
        if len(reactants_smiles) > 3:
            return False
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r) is not None]
            
            if not product_mol or not reactant_mols:
                return False
            
            product_atoms = product_mol.GetNumAtoms()
            reactant_atoms = [mol.GetNumAtoms() for mol in reactant_mols]
            max_reactant_atoms = max(reactant_atoms)
            
            # Linear reaction: largest reactant should be significant fraction of product
            # and there shouldn't be multiple large fragments being coupled
            if len([atoms for atoms in reactant_atoms if atoms > product_atoms * 0.3]) <= 1:
                return True
            
            # Additional check: if one reactant is much larger than others (>3x),
            # this suggests linear extension rather than convergent assembly
            sorted_reactants = sorted(reactant_atoms, reverse=True)
            if len(sorted_reactants) >= 2 and sorted_reactants[0] > 3 * sorted_reactants[1]:
                return True
                
        except Exception:
            return False
        
        return False
    
    def hit_condition(self, d) -> bool:
        """Not used in this implementation - using condition_depth instead"""
        return False
