"""Generated evaluation code for: Early alpha-beta unsaturated acid formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyUnsaturatedAcidFormation(BaseScoring):
    """
    Evaluates whether alpha-beta unsaturated acid formation via decarboxylation
    occurs early in the synthesis route (target step 7 out of 7 total steps).
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_number"]
        self.total_steps = config["parameters"]["total_steps"]
        self.target_depth_fraction = (self.total_steps - self.target_step) / self.total_steps
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        # Prefer early occurrence (close to target depth fraction)
        deviation = abs(x - self.target_depth_fraction)
        # Convert to 0-10 scale, with 10 being perfect timing
        return max(0, 10 * (1 - deviation * 2))
    
    def hit_condition(self, d):
        """Check if this reaction is a decarboxylation forming alpha-beta unsaturated acid"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Check if this is a decarboxylation (CO2 loss)
        if not self._is_decarboxylation(reactants_smiles, products_smiles):
            return False
            
        # Check if product contains alpha-beta unsaturated acid motif
        products = products_smiles.split(".")
        for prod_smiles in products:
            try:
                mol = Chem.MolFromSmiles(prod_smiles)
                if mol and self._has_unsaturated_acid(mol):
                    return True
            except:
                continue
                
        return False
    
    def _is_decarboxylation(self, reactants_smiles, products_smiles):
        """Check if reaction involves CO2 loss (decarboxylation)"""
        try:
            # Count carboxylic acid groups in reactants vs products
            reactant_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(s.strip()) for s in products_smiles.split(".")]
            
            carboxylic_pattern = Chem.MolFromSmarts("C(=O)[OH]")
            
            reactant_cooh_count = sum(len(mol.GetSubstructMatches(carboxylic_pattern)) 
                                    for mol in reactant_mols if mol)
            product_cooh_count = sum(len(mol.GetSubstructMatches(carboxylic_pattern)) 
                                   for mol in product_mols if mol)
            
            # Decarboxylation should reduce COOH count
            return reactant_cooh_count > product_cooh_count
            
        except:
            return False
    
    def _has_unsaturated_acid(self, mol):
        """Check if molecule contains alpha-beta unsaturated carboxylic acid (C=C-COOH)"""
        if not mol:
            return False
            
        try:
            # Pattern for alpha-beta unsaturated carboxylic acid
            # C=C connected to C(=O)OH
            pattern = Chem.MolFromSmarts("C=C-C(=O)[OH]")
            return mol.HasSubstructMatch(pattern)
        except:
            return False
