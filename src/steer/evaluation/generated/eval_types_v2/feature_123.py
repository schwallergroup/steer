"""Generated evaluation code for: Late stage Boc deprotection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocDeprotectionStrategy(BaseScoring):
    """
    Evaluates whether Boc deprotection occurs in the final step of synthesis.
    Checks for the presence of Boc-protected nitrogen that gets deprotected
    in the last reaction step.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "final_step")
        self.target_depth = 0 if self.timing == "final_step" else config.get("target_depth", 0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Boc deprotection doesn't happen
        elif self.timing == "final_step":
            # Perfect score if deprotection happens at depth 0 (final step)
            # Score decreases as depth increases
            return max(0, 1 - x) * 10
        else:
            # Score based on how close to target depth
            return max(0, 1 - abs(x - self.target_depth)) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Boc deprotection of nitrogen.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Boc protecting group pattern: tert-butoxycarbonyl on nitrogen
            boc_pattern = Chem.MolFromSmarts("[NX3,NX2][C](=O)OC(C)(C)C")
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
            
            if not boc_pattern:
                return False
            
            # Check if reactants contain Boc-protected nitrogen
            has_boc_reactant = any(mol.HasSubstructMatch(boc_pattern) for mol in reactant_mols)
            
            if not has_boc_reactant:
                return False
            
            # Check if products have free amine (indicating deprotection occurred)
            # or if Boc group is absent in products
            has_free_amine_product = any(mol.HasSubstructMatch(free_amine_pattern) for mol in product_mols)
            has_boc_product = any(mol.HasSubstructMatch(boc_pattern) for mol in product_mols)
            
            # Also check for tert-butanol or CO2 as byproducts (common in Boc deprotection)
            tbutanol_pattern = Chem.MolFromSmarts("OC(C)(C)C")
            co2_pattern = Chem.MolFromSmarts("O=C=O")
            
            has_deprotection_byproducts = any(
                mol.HasSubstructMatch(tbutanol_pattern) or mol.HasSubstructMatch(co2_pattern)
                for mol in product_mols
                if tbutanol_pattern and co2_pattern
            )
            
            # Boc deprotection occurred if:
            # 1. Reactant has Boc group
            # 2. Product has free amine OR lacks Boc group
            # 3. Optionally has deprotection byproducts
            return has_boc_reactant and (has_free_amine_product or not has_boc_product or has_deprotection_byproducts)
            
        except Exception:
            return False
