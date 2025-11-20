"""Generated evaluation code for: Late tetracycle formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateTetracycleFormation(BaseScoring):
    """
    Evaluates whether a tetracycle is formed late in the synthesis via cyclization
    that creates 2 rings simultaneously.
    """
    
    def __init__(self, config: Dict):
        self.target_rings = config["parameters"]["rings_formed"]
        self.stage = config["parameters"]["stage"]  # "late"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        if self.stage == "late":
            # Late-stage formation is better - reward higher depth fractions
            return x * 10  # Scale to 0-10 range
        else:
            # Early-stage formation would be (1-x) * 10
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms the target number of rings via cyclization"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactants + products):
                return False
            
            # Count rings in reactants and products
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
            
            rings_formed = product_rings - reactant_rings
            
            # Check if we formed the target number of rings
            if rings_formed != self.target_rings:
                return False
            
            # Additional check: ensure this is a cyclization (intramolecular)
            # For cyclization, we expect same number of molecules but with ring formation
            if len(reactants) == len(products):
                # Check if any product has significantly more rings than corresponding reactant
                for prod in products:
                    prod_ring_count = prod.GetRingInfo().NumRings()
                    # Look for a reactant that could correspond to this product
                    for react in reactants:
                        react_ring_count = react.GetRingInfo().NumRings()
                        if prod_ring_count - react_ring_count >= self.target_rings:
                            return True
            
            # Alternative: check if main molecule gained rings (intermolecular cyclization)
            if rings_formed == self.target_rings:
                return True
                
            return False
            
        except Exception:
            return False
