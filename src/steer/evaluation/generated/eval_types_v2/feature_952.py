"""Generated evaluation code for: Late thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThiazoleRingFormation(BaseScoring):
    """
    Evaluates when thiazole ring formation occurs in the synthesis route.
    Rewards early formation when stage='early', late formation when stage='late'.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.stage = config["parameters"]["stage"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.stage == "early":
            return x  # Higher score for earlier formation (higher depth fraction)
        elif self.stage == "late":
            return 1 - x  # Higher score for later formation (lower depth fraction)
        else:
            return 0.5  # Neutral if stage not specified
    
    def hit_condition(self, d) -> bool:
        """Check if thiazole ring is formed in this reaction step."""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse reactants and product
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if r.strip()]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains thiazole ring
            product_has_thiazole = product_mol.HasSubstructMatch(self.ring_pattern)
            
            # Check if any reactant already contains thiazole ring
            reactants_have_thiazole = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
            
            # Ring formation occurs if product has thiazole but reactants don't
            return product_has_thiazole and not reactants_have_thiazole
            
        except Exception:
            return False
