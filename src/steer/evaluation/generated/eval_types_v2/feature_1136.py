"""Generated evaluation code for: Cbz protecting group for piperidine nitrogen"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzPiperidineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of Cbz (carboxybenzyl) protecting group 
    on piperidine nitrogen, with deprotection occurring in the final step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
        # SMARTS patterns for Cbz-protected piperidine
        self.cbz_piperidine_pattern = "[#6]1-[#6]-[#7](-[#6](=[#8])-[#8]-[#6]-c2ccccc2)-[#6]-[#6]-[#6]-1"
        self.free_piperidine_pattern = "[#6]1-[#6]-[#7H]-[#6]-[#6]-[#6]-1"
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return max(0, 1 - abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves Cbz deprotection of piperidine nitrogen
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants, products = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smi in reactants.split("."):
                mol = Chem.MolFromSmiles(r_smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smi in products.split("."):
                mol = Chem.MolFromSmiles(p_smi)
                if mol:
                    product_mols.append(mol)
            
            # Check if reactants contain Cbz-protected piperidine
            cbz_pattern = Chem.MolFromSmarts(self.cbz_piperidine_pattern)
            free_pattern = Chem.MolFromSmarts(self.free_piperidine_pattern)
            
            has_cbz_reactant = any(mol.HasSubstructMatch(cbz_pattern) for mol in reactant_mols)
            has_free_product = any(mol.HasSubstructMatch(free_pattern) for mol in product_mols)
            
            # Check for Cbz byproducts (benzyl alcohol, CO2, etc.)
            cbz_byproduct_patterns = [
                "OCc1ccccc1",  # benzyl alcohol
                "O=C=O",       # CO2
                "[OH2]"        # water
            ]
            
            has_cbz_byproducts = False
            for pattern_smi in cbz_byproduct_patterns:
                pattern = Chem.MolFromSmarts(pattern_smi)
                if any(mol.HasSubstructMatch(pattern) for mol in product_mols):
                    has_cbz_byproducts = True
                    break
            
            # Cbz deprotection: Cbz-protected piperidine → free piperidine + byproducts
            return has_cbz_reactant and has_free_product and has_cbz_byproducts
            
        except Exception:
            return False
