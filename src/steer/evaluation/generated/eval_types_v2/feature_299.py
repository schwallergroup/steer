"""Generated evaluation code for: Final step urea formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinalStepUreaFormation(BaseScoring):
    """
    Evaluates whether urea formation occurs in the final step of the synthesis route.
    Checks for the reaction between an isocyanate and an amine to form a urea bond.
    """
    
    def __init__(self, config: Dict):
        self.reaction_type = config["parameters"]["reaction_type"]
        self.position = config["parameters"]["position"]
        self.reactants = config["parameters"]["reactants"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Urea formation doesn't happen in final step
        else:
            return 10  # Perfect score if it happens in final step
    
    def hit_condition(self, d):
        """Check if this reaction is urea formation between isocyanate and amine"""
        try:
            # Get the mapped reaction SMILES
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if we have isocyanate and amine reactants
            has_isocyanate = False
            has_amine = False
            
            # SMARTS patterns
            isocyanate_pattern = Chem.MolFromSmarts("[N]=[C]=[O]")  # N=C=O
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            urea_pattern = Chem.MolFromSmarts("[NH]-[C](=[O])-[NH]")  # Urea pattern
            
            # Check reactants for isocyanate and amine
            for reactant in reactants:
                if reactant.HasSubstructMatch(isocyanate_pattern):
                    has_isocyanate = True
                if reactant.HasSubstructMatch(amine_pattern):
                    has_amine = True
            
            # Check if product contains urea group
            has_urea_product = product.HasSubstructMatch(urea_pattern)
            
            # Must have both required reactants and form urea product
            return has_isocyanate and has_amine and has_urea_product
            
        except Exception:
            return False
    
    def condition_depth(self, d):
        """Override to only check the final step (depth 0)"""
        try:
            # Check if this is a leaf node (final step)
            if not d.get("children"):
                if self.hit_condition(d):
                    return True, 0  # Found in final step
            return False, -1  # Not found in final step
        except Exception:
            return False, -1
