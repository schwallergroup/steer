"""Generated evaluation code for: Early stage carboxylic acid protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyCarboxylicAcidProtection(BaseScoring):
    """
    Evaluates whether carboxylic acid protection with tert-butyl ester occurs early in the synthesis.
    Returns higher scores when the protection reaction happens at the specified early stage.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config.get("step_position", 1)
        self.protecting_group = config.get("protecting_group", "tert-butyl ester")
        
        # SMARTS patterns for carboxylic acid and tert-butyl ester
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[CX3](=O)[OH]")
        self.tert_butyl_ester_pattern = Chem.MolFromSmarts("[CX3](=O)OC(C)(C)C")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        
        # Convert depth fraction to step number (approximate)
        estimated_step = int(x * 10) + 1  # Assuming ~10 steps max
        
        # Score higher for earlier protection, with maximum at target step
        if estimated_step <= self.target_step:
            return 10  # Perfect early protection
        elif estimated_step <= self.target_step + 2:
            return 8   # Acceptable early protection
        elif estimated_step <= 5:
            return 5   # Moderate timing
        else:
            return 2   # Late protection, not ideal
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents carboxylic acid protection with tert-butyl ester.
        """
        try:
            mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
                
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if reactants contain carboxylic acid
            has_carboxylic_acid_reactant = any(
                mol.HasSubstructMatch(self.carboxylic_acid_pattern) 
                for mol in reactant_mols
            )
            
            # Check if products contain tert-butyl ester
            has_tert_butyl_ester_product = any(
                mol.HasSubstructMatch(self.tert_butyl_ester_pattern) 
                for mol in product_mols
            )
            
            # Also check for tert-butyl containing reagents (tert-butyl bromide, tert-butanol, etc.)
            tert_butyl_reagent_patterns = [
                Chem.MolFromSmarts("C(C)(C)C"),  # tert-butyl group
                Chem.MolFromSmarts("C(C)(C)CBr"), # tert-butyl bromide
                Chem.MolFromSmarts("C(C)(C)CO")   # tert-butanol
            ]
            
            has_tert_butyl_reagent = any(
                any(mol.HasSubstructMatch(pattern) for pattern in tert_butyl_reagent_patterns)
                for mol in reactant_mols
            )
            
            # Protection reaction: carboxylic acid + tert-butyl reagent -> tert-butyl ester
            return (has_carboxylic_acid_reactant and 
                    has_tert_butyl_ester_product and 
                    has_tert_butyl_reagent)
            
        except Exception:
            return False
