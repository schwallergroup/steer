"""Generated evaluation code for: Late stage nitrile to acetamide conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileToAcetamideConversion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile to acetamide conversion.
    Specifically looks for the conversion of a nitrile group (-CN) to an 
    N-acetylmethylamine group in the final steps of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
    
    def route_scoring(self, x) -> float:
        """
        Score based on how late in the synthesis the conversion occurs.
        Later conversions (lower depth values) are scored higher.
        """
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            # Late-stage conversion is better, so invert the depth
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves nitrile to acetamide conversion.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define patterns
            nitrile_pattern = Chem.MolFromSmarts("[#6]C#N")  # Nitrile group
            acetamide_pattern = Chem.MolFromSmarts("[#6]C(=O)N([CH3])[CH3]")  # N-acetylmethylamine
            
            if not nitrile_pattern or not acetamide_pattern:
                return False
            
            # Check if reactants contain nitrile
            has_nitrile_reactant = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
            
            # Check if products contain acetamide
            has_acetamide_product = any(mol.HasSubstructMatch(acetamide_pattern) for mol in products)
            
            # Check if reactants lack acetamide (to ensure it's being formed)
            lacks_acetamide_reactant = not any(mol.HasSubstructMatch(acetamide_pattern) for mol in reactants)
            
            # Reaction should convert nitrile to acetamide
            return has_nitrile_reactant and has_acetamide_product and lacks_acetamide_reactant
            
        except Exception:
            return False
