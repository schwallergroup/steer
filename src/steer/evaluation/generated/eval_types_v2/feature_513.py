"""Generated evaluation code for: Late stage Buchwald-Hartwig amination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BuchwaldHartwigFinalStep(BaseScoring):
    """
    Evaluates if Buchwald-Hartwig amination occurs as the final step in synthesis.
    Checks for C-N bond formation between an aryl halide and an amine in the last reaction.
    """
    
    def __init__(self, config: Dict):
        self.position = config.get("position", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            if self.position == "final_step":
                return 10 if x > 0.9 else x * 5  # High score for very late stage
            else:
                return 1 - x  # Earlier is better for other positions
    
    def hit_condition(self, d):
        """Check if this reaction is a Buchwald-Hartwig amination"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for aryl halide reactant (Ar-X where X = Cl, Br, I)
            aryl_halide_pattern = Chem.MolFromSmarts("[cH0,cH1,cH2]-[Cl,Br,I]")
            has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactants)
            
            # Check for amine reactant (primary or secondary amine)
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants)
            
            # Check for C-N bond formation in product
            aryl_amine_pattern = Chem.MolFromSmarts("[c]-[NH1,NH0]")
            has_aryl_amine_product = product.HasSubstructMatch(aryl_amine_pattern)
            
            # Additional check for typical Buchwald-Hartwig conditions
            # Look for palladium catalyst indicators (not always present in SMILES)
            # or phosphine ligands, but focus on reaction pattern
            
            return has_aryl_halide and has_amine and has_aryl_amine_product
            
        except Exception:
            return False
