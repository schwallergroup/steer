"""Generated evaluation code for: Late thiourea to guanidine conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThioureaToGuanidineConversion(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiourea to guanidine conversion.
    Detects reactions where a thiourea moiety (NC(=S)N) is converted to guanidine (NC(=N)N).
    Earlier conversions (lower depth) are scored higher.
    """
    
    def __init__(self, config: Dict):
        self.thiourea_smarts = config.get("substrate_smarts", "NC(=S)N")
        self.guanidine_smarts = "NC(=N)N"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            return 1 - x  # Earlier conversion is better (lower depth fraction)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction converts thiourea to guanidine"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            # Parse molecules
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
                
            # Create SMARTS patterns
            thiourea_pattern = Chem.MolFromSmarts(self.thiourea_smarts)
            guanidine_pattern = Chem.MolFromSmarts(self.guanidine_smarts)
            
            if not thiourea_pattern or not guanidine_pattern:
                return False
            
            # Check if any reactant has thiourea and product has guanidine
            has_thiourea_reactant = any(mol.HasSubstructMatch(thiourea_pattern) for mol in react_mols if mol)
            has_guanidine_product = prod_mol.HasSubstructMatch(guanidine_pattern)
            
            # Additional check: product should not have thiourea (complete conversion)
            has_thiourea_product = prod_mol.HasSubstructMatch(thiourea_pattern)
            
            return has_thiourea_reactant and has_guanidine_product and not has_thiourea_product
            
        except Exception:
            return False
