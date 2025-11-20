"""Generated evaluation code for: Late stage nitrile hydrolysis to ester"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileHydrolysis(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile hydrolysis to ester reactions.
    Rewards routes where nitrile groups are converted to esters in later synthetic steps.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Nitrile hydrolysis doesn't happen
        else:
            # Late-stage hydrolysis is better, score inversely with depth
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction step performs nitrile hydrolysis to ester"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check if reactant contains nitrile group
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p) for p in products if Chem.MolFromSmiles(p)]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Define patterns
            nitrile_pattern = Chem.MolFromSmarts("[C]#N")
            ester_pattern = Chem.MolFromSmarts("C(=O)O[C]")
            
            # Check for nitrile in reactants
            has_nitrile_reactant = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactant_mols)
            
            # Check for ester in products
            has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in product_mols)
            
            # Verify nitrile consumption (less nitriles in products than reactants)
            reactant_nitriles = sum(len(mol.GetSubstructMatches(nitrile_pattern)) for mol in reactant_mols)
            product_nitriles = sum(len(mol.GetSubstructMatches(ester_pattern)) for mol in product_mols)
            
            return has_nitrile_reactant and has_ester_product and reactant_nitriles > 0
            
        except Exception:
            return False
