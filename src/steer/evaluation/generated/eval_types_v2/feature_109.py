"""Generated evaluation code for: Late stage urea coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageUreaCoupling(BaseScoring):
    """
    Evaluates if urea bond formation occurs late in the synthesis route.
    Detects urea coupling reactions and scores based on depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Urea coupling doesn't happen
        else:
            if self.timing == "late":
                return 1 - x  # Late-stage coupling is better (higher score for higher depth fraction)
            else:
                return x  # Early-stage coupling is better
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves urea bond formation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if any(mol is None for mol in reactants + products):
                return False
            
            # Count urea groups in reactants and products
            urea_pattern = Chem.MolFromSmarts("[NX3][CX3](=[OX1])[NX3]")
            
            reactant_ureas = sum(len(mol.GetSubstructMatches(urea_pattern)) for mol in reactants)
            product_ureas = sum(len(mol.GetSubstructMatches(urea_pattern)) for mol in products)
            
            if self.direction == "formation":
                # Urea formation: more urea groups in products than reactants
                return product_ureas > reactant_ureas
            else:
                # Urea breaking: more urea groups in reactants than products
                return reactant_ureas > product_ureas
                
        except Exception:
            return False
