"""Generated evaluation code for: Late stage oxadiazolone ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageOxadiazoloneFormation(BaseScoring):
    """
    Evaluates whether oxadiazolone ring formation occurs at a late stage in the synthesis.
    Uses SMARTS pattern matching to detect 1,2,4-oxadiazol-5-one ring formation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Later formation is better, higher score for smaller depth fraction
            elif self.timing == "early":
                return x  # Earlier formation is better, higher score for larger depth fraction
            else:
                return 1 if x >= 0 else 0  # Just check if formation occurs
    
    def hit_condition(self, d):
        """
        Check if oxadiazolone ring formation occurs in this reaction step.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (failed parsing)
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Count oxadiazolone rings in reactants and products
            reactant_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                    for mol in reactant_mols)
            product_ring_count = sum(len(mol.GetSubstructMatches(self.ring_pattern)) 
                                   for mol in product_mols)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_ring_count > reactant_ring_count
            elif self.direction == "breaking":
                # Ring breaking: more rings in reactants than products
                return reactant_ring_count > product_ring_count
            else:
                # Any change in ring count
                return reactant_ring_count != product_ring_count
                
        except (KeyError, IndexError, AttributeError):
            return False
