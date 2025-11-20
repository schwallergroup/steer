"""Generated evaluation code for: Mid stage pyrazinopyridazine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrazinopyridazineRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for mid-stage pyrazinopyridazine ring formation.
    Checks if the specified fused heterocyclic ring is formed via cyclization
    at an appropriate depth (around depth 3 for mid-stage timing).
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1nnc2nccnc2c1"
        self.timing = config["parameters"]["timing"]  # "mid"
        self.target_depth = 3 if self.timing == "mid" else 2  # mid-stage target depth
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't occur
        
        # For mid-stage timing, penalize both early and late formation
        if self.timing == "mid":
            # Optimal at target_depth, penalty increases with distance
            depth_penalty = abs(x - self.target_depth) * 0.3
            return max(0, 1.0 - depth_penalty)
        else:
            # General case: earlier formation is better
            return max(0, 1.0 - x * 0.2)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms the pyrazinopyridazine ring.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) 
                           for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) 
                          for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Create pattern for pyrazinopyridazine
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
            
            # Check if ring is absent in reactants but present in products
            ring_in_reactants = any(mol.HasSubstructMatch(ring_pattern) 
                                  for mol in reactant_mols if mol)
            ring_in_products = any(mol.HasSubstructMatch(ring_pattern) 
                                 for mol in product_mols if mol)
            
            # Ring formation occurs if ring is formed (not present in reactants, present in products)
            return not ring_in_reactants and ring_in_products
            
        except Exception:
            return False
