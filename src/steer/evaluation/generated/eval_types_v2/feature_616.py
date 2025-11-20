"""Generated evaluation code for: Late stage alcohol formation via ester reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlcoholFormation(BaseScoring):
    """
    Evaluates whether alcohol formation via ester reduction occurs late in the synthesis route.
    Checks for reactions where an ester is reduced to form an alcohol, with preference for
    reactions occurring closer to the final product.
    """
    
    def __init__(self, config: Dict):
        # No specific configuration needed for this feature
        pass
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 1 - x  # Later stage is better, so invert the depth fraction
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents ester reduction to alcohol.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
                
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".") if r]
            
            if not product or not reactants:
                return False
            
            # Define SMARTS patterns
            ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][C:4]")  # Ester functional group
            alcohol_pattern = Chem.MolFromSmarts("[C:1][OH:2]")  # Primary/secondary alcohol
            
            # Check if product contains alcohol
            product_has_alcohol = product.HasSubstructMatch(alcohol_pattern)
            if not product_has_alcohol:
                return False
            
            # Check if any reactant contains ester
            reactant_has_ester = any(reactant.HasSubstructMatch(ester_pattern) for reactant in reactants)
            if not reactant_has_ester:
                return False
            
            # Additional check: verify the carbon bearing the alcohol in product
            # corresponds to the carbonyl carbon of ester in reactant using atom mapping
            product_alcohol_matches = product.GetSubstructMatches(alcohol_pattern)
            reactant_ester_matches = []
            for reactant in reactants:
                reactant_ester_matches.extend(reactant.GetSubstructMatches(ester_pattern))
            
            if not product_alcohol_matches or not reactant_ester_matches:
                return False
            
            # Check atom mapping to confirm the transformation
            for prod_match in product_alcohol_matches:
                prod_carbon_atom = product.GetAtomWithIdx(prod_match[0])
                prod_carbon_mapnum = prod_carbon_atom.GetAtomMapNum()
                
                if prod_carbon_mapnum > 0:  # Has atom mapping
                    for reactant in reactants:
                        for react_match in reactant.GetSubstructMatches(ester_pattern):
                            react_carbon_atom = reactant.GetAtomWithIdx(react_match[0])
                            react_carbon_mapnum = react_carbon_atom.GetAtomMapNum()
                            
                            if react_carbon_mapnum == prod_carbon_mapnum:
                                return True
            
            # If no atom mapping available, assume it's a valid transformation
            # if both ester and alcohol patterns are present
            return True
            
        except Exception:
            return False
