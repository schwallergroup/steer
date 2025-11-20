"""Generated evaluation code for: Final lactam formation via intramolecular amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FinalLactamFormation(BaseScoring):
    """
    Evaluates if a lactam ring is formed as the final step via intramolecular amide coupling.
    Checks for the presence of the specified lactam pattern being formed in the last reaction.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "final"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.lactam_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Lactam formation doesn't happen
        
        if self.timing == "final":
            # For final step formation, we want x to be close to 1.0 (last step)
            if x >= 0.9:  # Very close to final step
                return 10
            elif x >= 0.7:  # Late stage but not final
                return 5
            else:  # Too early
                return 2
        else:
            # For other timing preferences, could be extended
            return abs(1.0 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms the target lactam ring.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the lactam pattern
            product_has_lactam = product.HasSubstructMatch(self.lactam_pattern)
            
            if not product_has_lactam:
                return False
            
            # Check if any reactant already has the complete lactam pattern
            # If so, this is not a lactam formation reaction
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.lactam_pattern):
                    return False
            
            # Additional check for intramolecular cyclization:
            # Look for a single reactant that could cyclize to form the lactam
            for reactant in reactants:
                if self._could_form_lactam_by_cyclization(reactant, product):
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _could_form_lactam_by_cyclization(self, reactant, product):
        """
        Check if the reactant could plausibly cyclize to form the lactam in the product.
        This looks for the presence of both amine and carboxylic acid/ester functionalities
        in the reactant that could form an amide bond.
        """
        # Check for amine functionality
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        # Check for carboxylic acid or ester that could form amide
        acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH,OR]")
        
        has_amine = reactant.HasSubstructMatch(amine_pattern)
        has_acid = reactant.HasSubstructMatch(acid_pattern)
        
        # For intramolecular cyclization, both functionalities should be present
        # in the same molecule
        return has_amine and has_acid
