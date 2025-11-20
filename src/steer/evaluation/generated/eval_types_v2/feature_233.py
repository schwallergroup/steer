"""Generated evaluation code for: Late stage piperidine ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring is formed late in the synthesis route.
    Returns higher scores when the target ring is formed closer to the final product.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.direction = config["parameters"]["direction"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late stage formation, lower depth fraction is better
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms the target ring structure"""
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        # Check if ring is formed in this step
        return self._is_ring_formed(product, reactants)
    
    def _is_ring_formed(self, product_smiles: str, reactant_smiles_list: list) -> bool:
        """Check if the target ring is present in product but not in any single reactant"""
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            reactant_mols = []
            for r_smiles in reactant_smiles_list:
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol is not None:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
                
            # Create pattern from SMARTS
            ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
            if ring_pattern is None:
                return False
                
            # Check if ring is present in product
            if not product_mol.HasSubstructMatch(ring_pattern):
                return False
                
            # Check if ring is already present in any single reactant
            for reactant_mol in reactant_mols:
                if reactant_mol.HasSubstructMatch(ring_pattern):
                    return False
                    
            # Ring is in product but not in any single reactant - ring formation occurred
            return True
            
        except Exception:
            return False
