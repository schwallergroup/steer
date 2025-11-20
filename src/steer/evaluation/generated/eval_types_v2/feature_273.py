"""Generated evaluation code for: Late stage benzyl ester deprotection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEsterDeprotection(BaseScoring):
    """
    Evaluates benzyl ester deprotection timing in synthesis routes.
    Checks if benzyl ester deprotection (revealing carboxylic acid) occurs
    at the final step via hydrogenolysis reaction patterns.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.method = config.get("method", "hydrogenolysis")
        
        # SMARTS patterns for benzyl ester and carboxylic acid
        self.benzyl_ester_pattern = Chem.MolFromSmarts("[C:1](=O)O[CH2]c1ccccc1")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C:1](=O)O")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Deprotection doesn't happen
        
        if self.timing == "final_step":
            # Reward very late stage deprotection (close to 1.0)
            if x > 0.9:
                return 10
            elif x > 0.7:
                return 7
            else:
                return 3
        else:
            # For other timing preferences, penalize distance from target
            return max(0, 10 - abs(x - 0.5) * 20)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction performs benzyl ester deprotection.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
                
            # Check if any reactant has benzyl ester
            has_benzyl_ester_reactant = any(
                mol.HasSubstructMatch(self.benzyl_ester_pattern) 
                for mol in reactants
            )
            
            # Check if any product has carboxylic acid
            has_carboxylic_acid_product = any(
                mol.HasSubstructMatch(self.carboxylic_acid_pattern)
                for mol in products
            )
            
            # Check for benzyl/toluene as leaving group (hydrogenolysis signature)
            toluene_pattern = Chem.MolFromSmarts("Cc1ccccc1")
            benzyl_alcohol_pattern = Chem.MolFromSmarts("OCc1ccccc1")
            
            has_benzyl_leaving_group = any(
                mol.HasSubstructMatch(toluene_pattern) or 
                mol.HasSubstructMatch(benzyl_alcohol_pattern)
                for mol in products
            )
            
            # Additional check for hydrogenolysis conditions (H2 presence)
            has_hydrogen = any("H" in Chem.MolToSmiles(mol) for mol in reactants)
            
            # Confirm this is benzyl ester deprotection
            is_deprotection = (has_benzyl_ester_reactant and 
                             has_carboxylic_acid_product and
                             has_benzyl_leaving_group)
            
            return is_deprotection
            
        except Exception:
            return False
