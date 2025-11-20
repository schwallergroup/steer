"""Generated evaluation code for: Mid-stage pyrazolo[1,5-a]pyridine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazoloPyridineFormation(BaseScoring):
    """
    Evaluates mid-stage pyrazolo[1,5-a]pyridine ring formation.
    Checks if the specified fused heterocycle is formed at the target depth.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["ring_smarts"]  # "n1ncc2ccccc21"
        self.timing = config["timing"]  # "mid"
        self.target_depth = config["step_depth"]  # 4
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "mid":
            # Penalize deviation from target depth
            depth_penalty = abs(x - self.target_depth / 8.0)  # Normalize to 0-1
            return max(0, 1 - depth_penalty * 2)  # Scale to 0-1, prefer exact timing
        else:
            return 1 - x  # Default: earlier is better
    
    def hit_condition(self, d) -> bool:
        """Check if pyrazolo[1,5-a]pyridine ring is formed in this reaction"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Check reactants don't have the ring
        reactant_mols = []
        for r_smiles in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactant_mols.append(mol)
        
        has_ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) 
                                   for mol in reactant_mols)
        
        # Check products have the ring
        product_mols = []
        for p_smiles in products.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is not None:
                product_mols.append(mol)
        
        has_ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) 
                                  for mol in product_mols)
        
        # Ring formation occurs if absent in reactants but present in products
        return not has_ring_in_reactants and has_ring_in_products
