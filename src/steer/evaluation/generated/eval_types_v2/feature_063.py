"""Generated evaluation code for: Late stage C-S bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCBondFormation(BaseScoring):
    """
    Evaluates routes based on late-stage C-S bond formation between aromatic carbons and sulfur.
    Checks for the formation of [c]-[S] bonds and rewards routes where this occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # C-S bond formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better (higher score for smaller depth fraction)
            else:
                return x  # Earlier formation is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves C-S bond formation by comparing
        the number of [c]-[S] bonds in products vs reactants.
        """
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        reactants = []
        for r_smiles in reactants_smiles.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactants.append(mol)
        
        products = []
        for p_smiles in products_smiles.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is not None:
                products.append(mol)
        
        # Count C-S bonds in reactants
        reactant_cs_bonds = 0
        for mol in reactants:
            matches = mol.GetSubstructMatches(self.bond_pattern)
            reactant_cs_bonds += len(matches)
        
        # Count C-S bonds in products
        product_cs_bonds = 0
        for mol in products:
            matches = mol.GetSubstructMatches(self.bond_pattern)
            product_cs_bonds += len(matches)
        
        # Check for bond formation (more C-S bonds in products than reactants)
        if self.direction == "formation":
            return product_cs_bonds > reactant_cs_bonds
        elif self.direction == "breaking":
            return reactant_cs_bonds > product_cs_bonds
        else:
            # Any change in C-S bond count
            return product_cs_bonds != reactant_cs_bonds
