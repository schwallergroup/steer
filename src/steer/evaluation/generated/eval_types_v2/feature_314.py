"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs at a late stage in the synthesis route.
    Checks for amide bond formation reactions and scores based on their timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config["parameters"].get("step_position", 1)
        
    def route_scoring(self, x) -> float:
        """
        Score based on how late the amide coupling occurs.
        x is the depth fraction where amide coupling happens.
        Later stages (higher x values) get better scores.
        """
        if x < 0:
            return 0  # No amide coupling found
        
        # For late stage coupling, we want higher depth fractions
        # Scale to 0-10 range with late stage preferred
        return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents an amide coupling reaction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
            # Check if amide bond is formed in this reaction
            return self._is_amide_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _is_amide_coupling(self, product, reactants) -> bool:
        """
        Detect if this reaction represents amide bond formation.
        """
        # Define amide bond pattern
        amide_pattern = Chem.MolFromSmarts("[C](=[O])-[N]")
        
        if not amide_pattern:
            return False
            
        # Count amide bonds in product
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        
        # Count total amide bonds in all reactants
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        # Amide coupling if product has more amide bonds than sum of reactants
        if product_amides > reactant_amides:
            return True
            
        # Additional check: look for typical amide coupling patterns
        # Carboxylic acid or ester + amine patterns
        acid_pattern = Chem.MolFromSmarts("[C](=[O])-[O]")
        amine_pattern = Chem.MolFromSmarts("[N;H1,H2]")
        
        has_acid_reactant = any(r.HasSubstructMatch(acid_pattern) for r in reactants)
        has_amine_reactant = any(r.HasSubstructMatch(amine_pattern) for r in reactants)
        
        return has_acid_reactant and has_amine_reactant and product_amides > 0
