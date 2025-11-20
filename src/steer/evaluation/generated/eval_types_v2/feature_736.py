"""Generated evaluation code for: Late oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateOxadiazoloneFormation(BaseScoring):
    """
    Evaluates whether oxadiazolone ring formation occurs late in the synthesis route.
    
    Checks for the formation of 1,2,4-oxadiazol-5-one rings and rewards routes where
    this ring formation happens in later stages of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        For late formation, higher depth fractions get better scores.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late formation is rewarded - closer to 1.0 gets higher score
            if self.timing == "late":
                return 10 * x  # Linear scaling favoring late formation
            else:
                return 10 * (1 - x)  # Early formation
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node involves oxadiazolone ring formation.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Count oxadiazolone rings in reactants and products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                               for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                              for mol in products)
            
            # Check for ring formation (more rings in products than reactants)
            if self.direction == "formation":
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                return reactant_rings > product_rings
            else:
                return product_rings != reactant_rings
                
        except Exception:
            return False
