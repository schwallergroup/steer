"""Generated evaluation code for: Early stage quinoline core construction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class QuinolineCoreConstruction(BaseScoring):
    """
    Evaluates if quinoline core construction occurs early in the synthesis route.
    Rewards early-stage quinoline formation (before stage_threshold) via ring-forming reactions.
    """
    
    def __init__(self, config: Dict):
        self.quinoline_pattern = Chem.MolFromSmarts(config["parameters"]["ring_smarts"])
        self.timing = config["parameters"]["timing"]
        self.stage_threshold = config["parameters"]["stage_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Quinoline formation not detected
        
        if self.timing == "early":
            if x <= self.stage_threshold:
                return 10  # Perfect score for early formation
            else:
                # Linear decay for later formation
                return max(0, 10 * (1 - x) / (1 - self.stage_threshold))
        else:
            # For late-stage preference (if needed)
            if x >= self.stage_threshold:
                return 10
            else:
                return max(0, 10 * x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction forms a quinoline ring by comparing
        reactants and products for quinoline substructure presence.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            # Parse products
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol:
                    product_mols.append(mol)
            
            # Check if quinoline is absent in reactants but present in products
            quinoline_in_reactants = any(mol.HasSubstructMatch(self.quinoline_pattern) 
                                       for mol in reactant_mols)
            quinoline_in_products = any(mol.HasSubstructMatch(self.quinoline_pattern) 
                                      for mol in product_mols)
            
            # Return True if quinoline ring is formed in this step
            return not quinoline_in_reactants and quinoline_in_products
            
        except Exception:
            return False
