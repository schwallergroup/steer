"""Generated evaluation code for: Late stage nitrile to aldehyde reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileReduction(BaseScoring):
    """
    Evaluates routes for late-stage nitrile to aldehyde reduction reactions.
    Checks if the final step involves converting a nitrile group to an aldehyde.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 1.0)

    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0
            # For late-stage reactions, prefer depth closer to 1.0 (final steps)
            return max(0, 1 - abs(x - self.target_depth))

    def hit_condition(self, d):
        """
        Check if this reaction involves nitrile to aldehyde reduction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Define nitrile and aldehyde patterns
            nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")  # C≡N
            aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)[#6]")  # Aldehyde carbon
            
            # Check if any reactant has nitrile and any product has aldehyde
            has_nitrile_reactant = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
            has_aldehyde_product = any(mol.HasSubstructMatch(aldehyde_pattern) for mol in products)
            
            # Additional check: ensure nitrile is actually being reduced
            if has_nitrile_reactant and has_aldehyde_product:
                # Count nitriles in reactants and aldehydes in products
                reactant_nitriles = sum(len(mol.GetSubstructMatches(nitrile_pattern)) for mol in reactants)
                product_aldehydes = sum(len(mol.GetSubstructMatches(aldehyde_pattern)) for mol in products)
                
                # Check if we're actually converting nitrile to aldehyde (not just coincidental presence)
                if reactant_nitriles > 0 and product_aldehydes > 0:
                    return True
            
            return False
            
        except Exception:
            return False
