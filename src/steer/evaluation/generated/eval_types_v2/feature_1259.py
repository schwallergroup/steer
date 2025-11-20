"""Generated evaluation code for: Late stage amine protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmineProtection(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Boc protection of secondary amines.
    Rewards routes where Boc protection occurs at the specified depth (typically 0 for final step).
    """
    
    def __init__(self, config: Dict):
        self.target_depth = config["parameters"]["depth"]
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        else:
            # Perfect score if at target depth, decreases with distance
            depth_penalty = abs(x - self.target_depth) * 0.1
            return max(0, 1.0 - depth_penalty)
    
    def hit_condition(self, d):
        """Check if this reaction involves Boc protection of a secondary amine"""
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for Boc reagent in reactants
            boc_patterns = [
                "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",  # Boc2O
                "CC(C)(C)OC(=O)Cl",  # Boc-Cl
                "CC(C)(C)OC(=O)ON1C(=O)CCC1=O"  # Boc-OSu
            ]
            
            has_boc_reagent = any(
                any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                    for mol in reactant_mols if mol)
                for pattern in boc_patterns
            )
            
            if not has_boc_reagent:
                return False
            
            # Check for secondary amine in reactants and Boc-protected amine in products
            secondary_amine_pattern = "[NH1]([CH])[CH]"  # Secondary amine
            boc_protected_pattern = "[NH1]C(=O)OC(C)(C)C"  # Boc-protected amine
            
            has_secondary_amine = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(secondary_amine_pattern))
                for mol in reactant_mols if mol
            )
            
            has_boc_protected = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(boc_protected_pattern))
                for mol in product_mols if mol
            )
            
            return has_boc_reagent and has_secondary_amine and has_boc_protected
            
        except Exception:
            return False
