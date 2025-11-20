"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoleRingFormation(BaseScoring):
    """
    Evaluates synthesis routes based on the timing of pyrazole ring formation.
    Detects when a specific pyrazole substructure is formed and scores based 
    on whether it occurs late in the synthesis sequence.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" or "early"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x: float) -> float:
        """
        Convert depth fraction to score (0-10).
        For late formation: higher depth fraction is better.
        For early formation: lower depth fraction is better.
        """
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        if self.timing == "late":
            return x * 10  # Late formation gets higher score
        else:  # early
            return (1 - x) * 10  # Early formation gets higher score
    
    def hit_condition(self, d: Dict) -> bool:
        """
        Check if this reaction involves pyrazole ring formation/breaking.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check for pyrazole pattern in reactants and products
            reactants_have_pyrazole = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            products_have_pyrazole = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            
            if self.direction == "formation":
                # Ring formation: absent in reactants, present in products
                return not reactants_have_pyrazole and products_have_pyrazole
            else:  # direction == "break"
                # Ring breaking: present in reactants, absent in products
                return reactants_have_pyrazole and not products_have_pyrazole
                
        except Exception:
            return False
