"""Generated evaluation code for: Late stage esterification"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterification(BaseScoring):
    """
    Evaluates whether esterification occurs as a late-stage transformation.
    Detects carboxylic acid to ester conversion and rewards later occurrence.
    """
    
    def __init__(self, config: Dict):
        # No specific config needed for this implementation
        pass

    def route_scoring(self, x) -> float:
        """
        Score based on how late the esterification occurs.
        Later esterification (higher x) gets better score.
        """
        if x < 0:
            return 0  # No esterification found
        else:
            return x * 10  # Late-stage gets higher score (closer to 10)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves esterification.
        Detects carboxylic acid to ester transformation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define patterns for carboxylic acid and ester
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O;!H]")
            
            # Check if we have carboxylic acid in reactants and ester in products
            has_acid_reactant = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in reactant_mols)
            has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in product_mols)
            
            # Additional check: ensure we're not just hydrolyzing an ester
            has_ester_reactant = any(mol.HasSubstructMatch(ester_pattern) for mol in reactant_mols)
            has_acid_product = any(mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in product_mols)
            
            # True esterification: acid -> ester, not ester -> acid
            return has_acid_reactant and has_ester_product and not (has_ester_reactant and has_acid_product)
            
        except Exception:
            return False
