"""Generated evaluation code for: Late stage nitro reduction and protection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitroReductionProtection(BaseScoring):
    """
    Evaluates routes for late-stage nitro reduction combined with amine protection.
    Checks if nitro group reduction occurs in the final stages and is followed by
    or combined with Boc protection in a one-pot strategy.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("late_stage_threshold", 0.8)  # Consider last 20% as late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Required reaction combination not found
        # Late stage is better, so higher depth fraction gets higher score
        return x * 10
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves nitro reduction with amine protection"""
        metadata = d.get("metadata", {})
        rxn_smiles = metadata.get("mapped_reaction_smiles", "")
        
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        # Split reaction SMILES
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        try:
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
                
            # Check for nitro reduction: nitro group in reactants, amine in products
            nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")  # Nitro group
            primary_amine_pattern = Chem.MolFromSmarts("[NH2]")  # Primary amine
            boc_amine_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")  # Boc-protected amine
            
            # Check if reactants contain nitro group
            has_nitro_reactant = any(mol.HasSubstructMatch(nitro_pattern) for mol in reactants if mol)
            
            # Check if products contain Boc-protected amine (indicating protection step)
            has_boc_product = any(mol.HasSubstructMatch(boc_amine_pattern) for mol in products if mol)
            
            # Alternative: check if primary amine is formed (for separate protection step)
            has_amine_product = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in products if mol)
            
            # Also check for Boc reagent in reactants (indicating protection in same pot)
            boc_reagent_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)ON1C(=O)CCC1=O")  # Boc-ONSu
            boc2o_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")  # Boc2O
            
            has_boc_reagent = any(
                mol.HasSubstructMatch(boc_reagent_pattern) or mol.HasSubstructMatch(boc2o_pattern) 
                for mol in reactants if mol
            )
            
            # Return True if we have nitro reduction with protection
            # Either: nitro → Boc-amine (one pot) or nitro → amine with Boc reagent present
            return (has_nitro_reactant and has_boc_product) or \
                   (has_nitro_reactant and has_amine_product and has_boc_reagent)
                   
        except Exception:
            return False
