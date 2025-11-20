"""Generated evaluation code for: p-nitrobenzyl ester carboxylic acid protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PNitrobenzylEsterProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of p-nitrobenzyl ester protection 
    of carboxylic acids. Checks for the formation or cleavage of p-nitrobenzyl 
    ester bonds at various depths in the synthesis tree.
    """
    
    def __init__(self, config):
        self.present = config["parameters"]["present"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
        # SMARTS pattern for p-nitrobenzyl ester
        self.pnb_ester_pattern = "[#6](=O)O[CH2]c1ccc(cc1)[N+](=O)[O-]"
        # Pattern for free carboxylic acid
        self.carboxylic_acid_pattern = "[#6](=O)[OH]"
        
    def route_scoring(self, x):
        if x < 0:
            # Condition not met in route
            return 0 if self.present else 1
        else:
            # Condition met - earlier is generally better for protection
            if self.present:
                return 1 - x  # Reward early protection
            else:
                return x  # Penalize if protection found but not wanted
                
    def hit_condition(self, d):
        """
        Check if this reaction involves p-nitrobenzyl ester protection/deprotection
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
                
            # Count p-nitrobenzyl esters in reactants and products
            pnb_pattern = Chem.MolFromSmarts(self.pnb_ester_pattern)
            carbox_pattern = Chem.MolFromSmarts(self.carboxylic_acid_pattern)
            
            if pnb_pattern is None or carbox_pattern is None:
                return False
                
            reactant_pnb_count = sum(len(mol.GetSubstructMatches(pnb_pattern)) 
                                   for mol in reactants if mol is not None)
            product_pnb_count = sum(len(mol.GetSubstructMatches(pnb_pattern)) 
                                  for mol in products if mol is not None)
            
            reactant_carbox_count = sum(len(mol.GetSubstructMatches(carbox_pattern)) 
                                      for mol in reactants if mol is not None)
            product_carbox_count = sum(len(mol.GetSubstructMatches(carbox_pattern)) 
                                     for mol in products if mol is not None)
            
            # Check for protection (carboxylic acid -> p-nitrobenzyl ester)
            protection_occurred = (reactant_carbox_count > product_carbox_count and 
                                 product_pnb_count > reactant_pnb_count)
            
            # Check for deprotection (p-nitrobenzyl ester -> carboxylic acid)
            deprotection_occurred = (reactant_pnb_count > product_pnb_count and 
                                   product_carbox_count > reactant_carbox_count)
            
            return protection_occurred or deprotection_occurred
            
        except Exception:
            return False
