"""Generated evaluation code for: Late stage oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class OxadiazoloneRingFormation(BaseScoring):
    """
    Evaluates late-stage oxadiazolone ring formation in synthesis routes.
    Detects when an oxadiazolone ring ([#6]1[#7][#8][#6](=[#8])[#7]1) is formed
    and scores based on the timing of formation (later is better).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        else:
            # Late-stage formation is better, so invert the depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if oxadiazolone ring formation occurs in this reaction step.
        Returns True if the ring is absent in reactants but present in products.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles.strip())
                if mol is not None:
                    reactant_mols.append(mol)
            
            # Parse products  
            product_mols = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles.strip())
                if mol is not None:
                    product_mols.append(mol)
            
            # Check if oxadiazolone ring is absent in reactants
            ring_in_reactants = any(
                mol.HasSubstructMatch(self.ring_pattern) 
                for mol in reactant_mols
            )
            
            # Check if oxadiazolone ring is present in products
            ring_in_products = any(
                mol.HasSubstructMatch(self.ring_pattern)
                for mol in product_mols
            )
            
            # Ring formation: absent in reactants, present in products
            return not ring_in_reactants and ring_in_products
            
        except Exception:
            return False
