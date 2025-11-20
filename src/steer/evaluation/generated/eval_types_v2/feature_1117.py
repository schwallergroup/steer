"""Generated evaluation code for: Late pyrrolidine ring formation via double alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrrolidineFormationDepth(BaseScoring):
    """
    Evaluates synthesis routes based on the depth at which pyrrolidine ring formation occurs.
    Rewards late-stage pyrrolidine ring formation via double alkylation reactions.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]  # "C1CCCN1" for pyrrolidine
        self.target_formation_depth = config["formation_depth"]  # Expected depth (1 = very late)
        self.total_depth = config["total_depth"]  # Total synthesis depth for normalization
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        # Convert depth fraction to actual depth
        actual_depth = int(x * self.total_depth)
        
        # Reward formations close to target depth, with preference for late-stage
        depth_difference = abs(actual_depth - self.target_formation_depth)
        
        # Score inversely proportional to depth difference
        # Perfect match (depth_difference = 0) gives score of 1.0
        # Larger differences give lower scores
        if depth_difference == 0:
            return 1.0
        else:
            return max(0, 1.0 - (depth_difference / self.total_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a pyrrolidine ring via double alkylation.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains pyrrolidine ring
            if not product.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant contains the pyrrolidine ring
            # If so, this is not a ring formation reaction
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False
            
            # Additional check: look for double alkylation pattern
            # This would involve N-containing reactant and two alkyl leaving groups
            return self._is_double_alkylation(reactants, product)
            
        except Exception:
            return False
    
    def _is_double_alkylation(self, reactants, product) -> bool:
        """
        Check if the reaction pattern matches double N-alkylation.
        Look for nitrogen-containing reactant and alkylating agents.
        """
        try:
            # Look for nitrogen-containing starting material
            n_containing_reactants = []
            alkylating_agents = []
            
            for reactant in reactants:
                has_nitrogen = any(atom.GetSymbol() == 'N' for atom in reactant.GetAtoms())
                if has_nitrogen:
                    n_containing_reactants.append(reactant)
                else:
                    # Check for common leaving groups indicating alkylating agent
                    leaving_groups = ["[Br]", "[Cl]", "[I]", "OS(=O)(=O)C"]  # Br, Cl, I, OTs
                    for lg_smarts in leaving_groups:
                        lg_pattern = Chem.MolFromSmarts(lg_smarts)
                        if reactant.HasSubstructMatch(lg_pattern):
                            alkylating_agents.append(reactant)
                            break
            
            # For double alkylation: expect 1 N-containing reactant and 1-2 alkylating agents
            return len(n_containing_reactants) >= 1 and len(alkylating_agents) >= 1
            
        except Exception:
            return False
