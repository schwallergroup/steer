"""Generated evaluation code for: Furan to pyridine ring transformation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FuranToPyridineTransformation(BaseScoring):
    """
    Evaluates synthesis routes for furan to pyridine ring transformation timing.
    Checks when a furan ring (c1ccoc1) is converted to a pyridine ring (c1ccncc1)
    and scores based on the desired timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.starting_ring_smarts = config["parameters"]["starting_ring_smarts"]  # c1ccoc1
        self.product_ring_smarts = config["parameters"]["product_ring_smarts"]    # c1ccncc1
        self.timing = config["parameters"]["timing"]  # "mid"
        self.direction = config["parameters"]["direction"]  # "transformation"
        
        # Convert timing preference to target depth fraction
        if self.timing == "early":
            self.target_depth_fraction = 0.8  # Early in route (high depth fraction)
        elif self.timing == "mid":
            self.target_depth_fraction = 0.5  # Mid-stage
        elif self.timing == "late":
            self.target_depth_fraction = 0.2  # Late in route (low depth fraction)
        else:
            self.target_depth_fraction = 0.5  # Default to mid
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Transformation doesn't happen
        
        # Score based on how close the actual depth is to target timing
        deviation = abs(x - self.target_depth_fraction)
        # Convert to 0-10 scale where 0 deviation = 10 points
        score = max(0, 10 - (deviation * 20))  # Scale deviation to 0-10 range
        return score
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node contains a furan to pyridine transformation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
        except Exception:
            return False
        
        if not reactants or not products:
            return False
        
        # Create pattern molecules for substructure matching
        furan_pattern = Chem.MolFromSmarts(self.starting_ring_smarts)
        pyridine_pattern = Chem.MolFromSmarts(self.product_ring_smarts)
        
        if furan_pattern is None or pyridine_pattern is None:
            return False
        
        # Check for furan in reactants and pyridine in products
        has_furan_reactant = any(mol.HasSubstructMatch(furan_pattern) for mol in reactants)
        has_pyridine_product = any(mol.HasSubstructMatch(pyridine_pattern) for mol in products)
        
        # For transformation direction, we want furan disappearing and pyridine appearing
        if self.direction == "transformation":
            # Also check that pyridine is not in reactants and furan is not in products
            # to ensure it's a true transformation
            has_pyridine_reactant = any(mol.HasSubstructMatch(pyridine_pattern) for mol in reactants)
            has_furan_product = any(mol.HasSubstructMatch(furan_pattern) for mol in products)
            
            return (has_furan_reactant and has_pyridine_product and 
                   not has_pyridine_reactant and not has_furan_product)
        
        return False
