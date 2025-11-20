"""Generated evaluation code for: Linear synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LinearSynthesis(MultiRxnCondBase):
    """
    Evaluates if a synthesis route follows a linear strategy.
    
    A linear synthesis is characterized by having mostly single reactant to product 
    transformations, avoiding convergent coupling of pre-formed complex fragments.
    Scores routes based on how closely they follow a linear approach.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "linear")
        self.fragment_count = config.get("fragment_count", 1)
        self.convergent_penalty = config.get("convergent_penalty", 0.5)
        
    def condition_depth(self, d):
        """
        Analyzes the entire route tree to determine linearity.
        Returns (condition_met, linearity_score) where linearity_score 
        ranges from 0 (highly convergent) to 1 (perfectly linear).
        """
        reactions = self.get_rxns(d)
        
        if not reactions:
            return False, 0
            
        linear_score = self._calculate_linearity_score(reactions)
        
        # Condition is met if the route is sufficiently linear
        condition_met = linear_score >= 0.7
        
        return condition_met, linear_score
        
    def _calculate_linearity_score(self, reactions):
        """
        Calculate how linear the synthesis route is based on reaction patterns.
        """
        if not reactions:
            return 0.0
            
        convergent_reactions = 0
        total_reactions = len(reactions)
        
        for rxn in reactions:
            if self._is_convergent_reaction(rxn):
                convergent_reactions += 1
                
        # Linear score decreases with more convergent reactions
        linearity_ratio = 1.0 - (convergent_reactions / total_reactions)
        
        return max(0.0, linearity_ratio)
        
    def _is_convergent_reaction(self, rxn):
        """
        Determine if a reaction represents a convergent step.
        Convergent reactions typically involve coupling two or more 
        complex fragments of similar size.
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0].split(".")
            
            # Single reactant transformations are linear
            if len(reactants_smiles) <= 1:
                return False
                
            # Check if we have multiple complex reactants (convergent indicator)
            complex_reactants = 0
            
            for reactant_smi in reactants_smiles:
                mol = Chem.MolFromSmiles(reactant_smi)
                if mol and self._is_complex_fragment(mol):
                    complex_reactants += 1
                    
            # Convergent if 2+ complex fragments are coupled
            return complex_reactants >= 2
            
        except Exception:
            return False
            
    def _is_complex_fragment(self, mol):
        """
        Determine if a molecule represents a complex synthetic fragment.
        Uses molecular weight, ring count, and functional group diversity.
        """
        if mol is None:
            return False
            
        # Molecular weight threshold for complexity
        mw = Descriptors.MolWt(mol)
        if mw < 100:  # Simple building blocks
            return False
            
        # Ring-containing molecules are often complex fragments  
        ring_count = mol.GetRingInfo().NumRings()
        
        # Functional group patterns indicating synthetic complexity
        complex_patterns = [
            "[#6]([#8])([#8])",  # Protected carbonyls/alcohols
            "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1",  # Aromatic rings
            "[#7]([#6])([#6])",  # Tertiary amines
            "[#16](=[#8])(=[#8])",  # Sulfonyl groups
        ]
        
        functional_groups = 0
        for pattern in complex_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                functional_groups += 1
                
        # Fragment is complex if it has rings + functional groups or high MW
        return (ring_count > 0 and functional_groups > 0) or mw > 250
        
    def route_scoring(self, linearity_score):
        """
        Convert linearity score to 0-10 scale.
        Higher scores for more linear routes.
        """
        if linearity_score < 0:
            return 0
            
        # Scale linearity score (0-1) to scoring range (0-10)
        return linearity_score * 10
