"""Generated evaluation code for: Late stage electrophilic bromination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageElectrophilicBromination(BaseScoring):
    """
    Evaluates routes for late-stage electrophilic bromination reactions using NBS.
    Scores routes based on when electrophilic bromination occurs - later is better.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Later bromination (higher x) gets better score.
        """
        if x < 0:
            return 0  # Bromination doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x >= self.target_depth else 0
        else:
            # Reward late-stage bromination (x closer to 1.0)
            if x >= self.target_depth:
                return 10
            else:
                # Linear penalty for earlier bromination
                return 10 * (x / self.target_depth)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents electrophilic bromination with NBS.
        """
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for NBS reagent in reactants
            nbs_pattern = Chem.MolFromSmarts("[Br][N]1[CH2][CH2][CH2][CH2][C]1=O")  # NBS pattern
            nbs_alt_pattern = Chem.MolFromSmarts("[Br][N]([CH2][CH2][CH2][CH2][C]=O)")  # Alternative NBS
            
            has_nbs = any(
                mol.HasSubstructMatch(nbs_pattern) or mol.HasSubstructMatch(nbs_alt_pattern)
                for mol in reactants if mol is not None
            )
            
            if not has_nbs:
                return False
            
            # Check for bromination (Br addition to aromatic/aliphatic carbon)
            # Count bromine atoms in reactants vs products
            reactant_br_count = sum(
                len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
                for mol in reactants if mol is not None
            )
            
            product_br_count = sum(
                len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
                for mol in products if mol is not None
            )
            
            # In electrophilic bromination, Br should be incorporated into substrate
            # (net Br count in organic products should increase)
            organic_products = [mol for mol in products if mol is not None and 
                              not any(atom.GetSymbol() == 'N' and 
                                    mol.HasSubstructMatch(Chem.MolFromSmarts("[N]1[CH2][CH2][CH2][CH2][C]1=O"))
                                    for atom in mol.GetAtoms())]
            
            organic_product_br_count = sum(
                len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
                for mol in organic_products
            )
            
            # Check if bromine was added to organic substrate
            organic_reactants = [mol for mol in reactants if mol is not None and 
                               not (mol.HasSubstructMatch(nbs_pattern) or mol.HasSubstructMatch(nbs_alt_pattern))]
            
            organic_reactant_br_count = sum(
                len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br'])
                for mol in organic_reactants
            )
            
            return organic_product_br_count > organic_reactant_br_count
            
        except Exception:
            return False
