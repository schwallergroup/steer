"""Generated evaluation code for: Halodesilylation to aryl iodide strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HalodesilylationStrategy(BaseScoring):
    """
    Evaluates synthesis routes for halodesilylation reactions that break C-Si bonds 
    and form C-I bonds, converting aryl-trimethylsilyl substrates to aryl iodides.
    This strategy is useful for regioselective iodination prior to cross-coupling.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Halodesilylation doesn't happen
        else:
            # Earlier halodesilylation is generally better for synthetic strategy
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a halodesilylation by detecting:
        1. Loss of trimethylsilyl group (C-Si bond break)
        2. Formation of aryl iodide (C-I bond formation)
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for trimethylsilyl group in reactants
            tms_pattern = Chem.MolFromSmarts("[Si](C)(C)C")  # Trimethylsilyl group
            has_tms_reactant = any(mol.HasSubstructMatch(tms_pattern) for mol in reactants)
            
            # Check for aryl iodide in products
            aryl_iodide_pattern = Chem.MolFromSmarts("c-I")  # Aromatic carbon bonded to iodine
            has_aryl_iodide_product = any(mol.HasSubstructMatch(aryl_iodide_pattern) for mol in products)
            
            # Check that TMS is not present in products (confirming C-Si bond break)
            has_tms_product = any(mol.HasSubstructMatch(tms_pattern) for mol in products)
            
            # Additional check for aryl-Si bond in reactants
            aryl_si_pattern = Chem.MolFromSmarts("c-[Si]")  # Aromatic carbon bonded to silicon
            has_aryl_si_reactant = any(mol.HasSubstructMatch(aryl_si_pattern) for mol in reactants)
            
            # Halodesilylation conditions:
            # 1. Reactant has trimethylsilyl group attached to aromatic carbon
            # 2. Product has aryl iodide
            # 3. No TMS group remains in products
            return (has_tms_reactant and 
                   has_aryl_iodide_product and 
                   not has_tms_product and
                   has_aryl_si_reactant)
                   
        except Exception:
            return False
