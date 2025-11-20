"""Generated evaluation code for: Late stage aromatic decarboxylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAromaticDecarboxylation(BaseScoring):
    """
    Evaluates whether aromatic decarboxylation occurs at a late stage in the synthesis.
    Detects loss of carboxylic acid groups from aromatic rings and rewards when this
    occurs as a late-stage transformation.
    """
    
    def __init__(self, config: Dict):
        self.target_late_stage = config.get("late_stage_threshold", 0.8)  # Later than 80% of route
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Decarboxylation doesn't happen
        else:
            # Reward late-stage decarboxylation (higher depth fraction is better)
            if x >= self.target_late_stage:
                return 10.0  # Perfect score for very late stage
            else:
                return x * 10.0  # Scale depth fraction to 0-10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves aromatic decarboxylation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactant_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactant_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in product_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Pattern for aromatic carboxylic acid
            aromatic_carboxylic_pattern = Chem.MolFromSmarts("[cR]C(=O)O")
            
            # Check if any reactant has aromatic carboxylic acid
            has_aromatic_carboxylic_reactant = any(
                mol.HasSubstructMatch(aromatic_carboxylic_pattern) for mol in reactants
            )
            
            if not has_aromatic_carboxylic_reactant:
                return False
            
            # Check if products have lost the carboxylic acid group
            # Look for CO2 as a product or reduced carbon count
            co2_pattern = Chem.MolFromSmiles("O=C=O")
            has_co2_product = any(
                Chem.MolToSmiles(mol) == "O=C=O" for mol in products
            )
            
            # Alternative check: compare carbon counts to detect decarboxylation
            reactant_carbons = sum(mol.GetNumAtoms() for mol in reactants 
                                 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            product_carbons = sum(mol.GetNumAtoms() for mol in products
                                for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            
            # Decarboxylation should reduce carbon count by 1
            carbon_loss = reactant_carbons - product_carbons
            
            return has_co2_product or carbon_loss == 1
            
        except Exception:
            return False
