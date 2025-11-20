"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DiarylEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage diaryl ether formation.
    Detects when a diaryl ether bond (c-O-c) is formed in the synthetic route,
    with preference for formation occurring at later stages.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]  # "c-O-c"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diaryl ether formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later stage formation is better (higher score)
            else:
                return x  # Earlier stage formation is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if a diaryl ether bond is formed in this reaction step.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Create SMARTS pattern for diaryl ether
            diaryl_ether_pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not diaryl_ether_pattern:
                return False
            
            # Count diaryl ether bonds in reactants and products
            reactant_matches = sum(len(mol.GetSubstructMatches(diaryl_ether_pattern)) 
                                 for mol in reactants if mol is not None)
            product_matches = sum(len(mol.GetSubstructMatches(diaryl_ether_pattern)) 
                                for mol in products if mol is not None)
            
            # Check for bond formation (increase in diaryl ether count)
            if self.direction == "formation":
                return product_matches > reactant_matches
            elif self.direction == "breaking":
                return reactant_matches > product_matches
            else:
                return product_matches != reactant_matches
                
        except Exception:
            return False
