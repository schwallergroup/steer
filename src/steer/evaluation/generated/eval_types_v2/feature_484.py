"""Generated evaluation code for: TBDMS protecting group deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TBDMSDeprotection(BaseScoring):
    """
    Evaluates TBDMS protecting group deprotection strategy in synthesis routes.
    Detects when TBDMS silyl ether protecting groups are removed from alcohols.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # TBDMS deprotection doesn't occur
        else:
            if self.condition_type == "bool":
                return 1  # Condition met
            else:
                # Later deprotection is generally preferred
                return 1 - abs(x - self.target_depth)
    
    def hit_condition(self, d):
        """
        Check if a reaction involves TBDMS deprotection.
        Looks for TBDMS-protected alcohol in reactants and free alcohol in products.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # TBDMS-protected alcohol pattern: R-O-Si(C)(C)C(C)(C)(C)
            tbdms_pattern = Chem.MolFromSmarts("[OH0:1][Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]")
            
            # Check for TBDMS group in reactants
            has_tbdms_reactant = False
            tbdms_oxygen_maps = set()
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(tbdms_pattern):
                    has_tbdms_reactant = True
                    # Get atom map numbers of oxygens in TBDMS groups
                    matches = reactant.GetSubstructMatches(tbdms_pattern)
                    for match in matches:
                        oxygen_atom = reactant.GetAtomWithIdx(match[0])
                        if oxygen_atom.GetAtomMapNum() > 0:
                            tbdms_oxygen_maps.add(oxygen_atom.GetAtomMapNum())
            
            if not has_tbdms_reactant or not tbdms_oxygen_maps:
                return False
            
            # Check if corresponding free alcohols appear in products
            free_alcohol_pattern = Chem.MolFromSmarts("[OH1:1]")
            
            for product in products:
                if product.HasSubstructMatch(free_alcohol_pattern):
                    matches = product.GetSubstructMatches(free_alcohol_pattern)
                    for match in matches:
                        oxygen_atom = product.GetAtomWithIdx(match[0])
                        if oxygen_atom.GetAtomMapNum() in tbdms_oxygen_maps:
                            return True
            
            # Also check for TBDMS byproducts (like TBDMS-F, TBDMS-OH)
            tbdms_byproduct_patterns = [
                Chem.MolFromSmarts("[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]"),  # TBDMS fragment
                Chem.MolFromSmarts("F[Si]([CH3])([CH3])[C]([CH3])([CH3])[CH3]")   # TBDMS-F
            ]
            
            for product in products:
                for pattern in tbdms_byproduct_patterns:
                    if pattern and product.HasSubstructMatch(pattern):
                        return True
            
            return False
            
        except Exception:
            return False
