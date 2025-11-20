"""Generated evaluation code for: Late stage oxazinanone ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OxazinanoneRingClosure(BaseScoring):
    """
    Evaluates late-stage oxazinanone ring formation in synthesis routes.
    Detects formation of six-membered oxazinanone rings via intramolecular cyclization.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" for late-stage preference
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            # Late-stage formation is preferred (higher depth fraction is better)
            return x * 10  # Convert to 0-10 scale, favoring later stages
        else:
            # Early-stage formation preference
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms an oxazinanone ring."""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if product contains the oxazinanone ring
            product_has_ring = product.HasSubstructMatch(self.ring_pattern)
            if not product_has_ring:
                return False
            
            # Check if any reactant already contains the complete ring
            # (we want ring formation, not just ring preservation)
            for reactant in reactants:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not formation
            
            return True
            
        except (KeyError, ValueError, AttributeError):
            return False
