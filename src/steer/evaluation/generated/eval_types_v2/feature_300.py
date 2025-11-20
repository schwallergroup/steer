"""Generated evaluation code for: Linear synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearSynthesisApproach(BaseScoring):
    """
    Evaluates whether a synthesis route follows a linear approach by analyzing
    the branching factor and convergence patterns in the synthesis tree.
    Linear routes build complexity step-by-step with minimal convergent coupling.
    """
    
    def __init__(self, config: Dict):
        self.strategy_type = config["strategy_type"]  # "linear"
        self.fragment_count = config["fragment_count"]  # Expected fragments (1 for linear)
        self.late_coupling = config["late_coupling"]  # False for linear approach
        
    def route_scoring(self, x) -> float:
        """
        Score based on linearity fraction.
        x is the fraction of linear steps vs total steps.
        Higher x means more linear approach.
        """
        if x < 0:
            return 0  # No valid analysis possible
        
        # For linear strategy, higher linearity fraction is better
        if self.strategy_type == "linear":
            return x * 10  # Scale to 0-10, where 10 is fully linear
        else:
            return (1 - x) * 10  # For convergent strategies, lower linearity is better
    
    def condition_depth(self, d) -> Tuple[bool, float]:
        """
        Analyze the entire synthesis tree to determine linearity.
        Returns (condition_met, linearity_fraction)
        """
        # Collect all reaction nodes in the tree
        all_nodes = []
        self._collect_nodes(d, all_nodes)
        
        if len(all_nodes) == 0:
            return False, -1
        
        linear_steps = 0
        convergent_steps = 0
        
        for node in all_nodes:
            if self._is_linear_step(node):
                linear_steps += 1
            else:
                convergent_steps += 1
        
        total_steps = linear_steps + convergent_steps
        if total_steps == 0:
            return False, -1
        
        linearity_fraction = linear_steps / total_steps
        
        # Condition is met if route matches expected strategy
        condition_met = self._evaluate_strategy(linearity_fraction, len(all_nodes))
        
        return condition_met, linearity_fraction
    
    def _collect_nodes(self, node, all_nodes):
        """Recursively collect all reaction nodes in the tree"""
        if node.get("type") == "reaction":
            all_nodes.append(node)
        
        for child in node.get("children", []):
            self._collect_nodes(child, all_nodes)
    
    def _is_linear_step(self, node) -> bool:
        """
        Determine if a reaction step is linear (one major fragment + small reagents)
        vs convergent (multiple significant fragments coupling)
        """
        try:
            rxn_smiles = node["metadata"]["mapped_reaction_smiles"]
            prod_smiles, react_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in react_smiles.split(".")]
            reactants = [r for r in reactants if r is not None]
            
            if len(reactants) <= 1:
                return True  # Single reactant transformation is linear
            
            # Count significant fragments (>3 heavy atoms)
            significant_fragments = 0
            for reactant in reactants:
                heavy_atom_count = reactant.GetNumHeavyAtoms()
                if heavy_atom_count > 3:  # Threshold for significant fragment
                    significant_fragments += 1
            
            # Linear step: one major fragment + small reagents/catalysts
            # Convergent step: multiple significant fragments
            return significant_fragments <= 1
            
        except Exception:
            return True  # Default to linear if analysis fails
    
    def _evaluate_strategy(self, linearity_fraction, total_reactions) -> bool:
        """Evaluate if the route matches the expected strategy parameters"""
        
        if self.strategy_type == "linear":
            # For linear strategy, expect high linearity fraction
            linearity_threshold = 0.7
            
            # Check fragment count expectation
            fragment_condition = True
            if self.fragment_count == 1:
                # Expect minimal convergent steps
                fragment_condition = linearity_fraction >= linearity_threshold
            
            # Check late coupling expectation
            late_coupling_condition = True
            if not self.late_coupling:
                # Linear approach should not have late-stage coupling
                late_coupling_condition = linearity_fraction >= linearity_threshold
            
            return fragment_condition and late_coupling_condition
        
        return linearity_fraction >= 0.5  # Default threshold
