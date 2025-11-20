"""Generated evaluation code for: Nitrile intermediate in acid synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileIntermediateInAcidSynthesis(BaseScoring):
    """
    Evaluates synthesis routes that use nitrile as an intermediate step to form carboxylic acid.
    Checks for nitrile hydrolysis reactions where a nitrile group is converted to carboxylic acid.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", 0)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Nitrile hydrolysis doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Found nitrile hydrolysis
            else:
                # Earlier use of nitrile hydrolysis is better (closer to target)
                return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents nitrile hydrolysis to carboxylic acid"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define SMARTS patterns
            nitrile_pattern = Chem.MolFromSmarts("[C]#[N]")  # Nitrile group
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=[O])[OH]")  # Carboxylic acid
            
            # Check if any reactant contains nitrile
            has_nitrile_reactant = any(
                mol.HasSubstructMatch(nitrile_pattern) for mol in reactant_mols
            )
            
            # Check if any product contains carboxylic acid
            has_acid_product = any(
                mol.HasSubstructMatch(carboxylic_acid_pattern) for mol in product_mols
            )
            
            # Check that nitrile is consumed (not present in products)
            has_nitrile_product = any(
                mol.HasSubstructMatch(nitrile_pattern) for mol in product_mols
            )
            
            # Nitrile hydrolysis: nitrile in reactants, carboxylic acid in products, 
            # and nitrile not in products (or fewer nitrile groups in products)
            if has_nitrile_reactant and has_acid_product:
                # Count nitrile groups in reactants vs products
                reactant_nitriles = sum(
                    len(mol.GetSubstructMatches(nitrile_pattern)) for mol in reactant_mols
                )
                product_nitriles = sum(
                    len(mol.GetSubstructMatches(nitrile_pattern)) for mol in product_mols
                )
                
                # Nitrile hydrolysis occurred if we have fewer nitriles in products
                return reactant_nitriles > product_nitriles
            
            return False
            
        except Exception:
            return False
