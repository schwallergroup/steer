"""Generated evaluation code for: Early quinolinone core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyQuinolinoneFormation(BaseScoring):
    """
    Evaluates whether quinolinone core formation occurs early in the synthesis route.
    Checks for the formation of the quinolinone heterocycle (c1ccc2[nH]c(=O)ccc2c1)
    and rewards early formation in the synthetic sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.formation_step = config["parameters"]["formation_step"]
        self.total_steps = config["parameters"]["total_steps"]
        self.timing = config["parameters"]["timing"]
        self.quinolinone_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        # Convert depth fraction to step number
        actual_step = x * self.total_steps
        
        if self.timing == "early":
            # Reward early formation, penalize late formation
            if actual_step <= self.formation_step:
                return 10  # Perfect score for early formation
            else:
                # Linear penalty for late formation
                penalty = (actual_step - self.formation_step) / (self.total_steps - self.formation_step)
                return max(0, 10 * (1 - penalty))
        
        return 0
    
    def hit_condition(self, d) -> bool:
        """
        Check if quinolinone ring formation occurs in this reaction step.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            # Check if quinolinone is absent in reactants but present in products
            quinolinone_in_reactants = any(mol.HasSubstructMatch(self.quinolinone_pattern) for mol in reactants)
            quinolinone_in_products = any(mol.HasSubstructMatch(self.quinolinone_pattern) for mol in products)
            
            # Ring formation: absent in reactants, present in products
            return not quinolinone_in_reactants and quinolinone_in_products
            
        except Exception:
            return False
