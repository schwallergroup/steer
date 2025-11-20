"""Generated evaluation code for: Late stage nitrile installation via cyanation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitrileInstallation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitrile installation via cyanation reactions.
    Detects formation of C-CN bonds and rewards routes where this occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "[#6]-[#7]#[#6]"
        self.timing = config["parameters"]["timing"]  # "late"
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late-stage timing, later installation is better.
        """
        if x < 0:
            return 0  # Nitrile installation doesn't happen
        else:
            return 1 - x  # Later installation gets higher score
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents nitrile installation via cyanation.
        Returns True if a C-CN bond is formed in this step.
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
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactants.append(mol)
                    
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Create nitrile substructure pattern
            nitrile_pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not nitrile_pattern:
                return False
            
            # Check if nitrile is formed (present in products but not in reactants)
            nitrile_in_reactants = any(mol.HasSubstructMatch(nitrile_pattern) for mol in reactants)
            nitrile_in_products = any(mol.HasSubstructMatch(nitrile_pattern) for mol in products)
            
            # Nitrile installation: nitrile appears in products but not in reactants
            if nitrile_in_products and not nitrile_in_reactants:
                return True
                
            return False
            
        except Exception:
            return False
