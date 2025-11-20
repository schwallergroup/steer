"""Generated evaluation code for: Thioacetate protecting group for thiol installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThioacetateProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of thioacetate protecting groups for thiol installation.
    
    This class checks if thioacetate groups are used as protecting groups for thiols,
    which prevents oxidation to disulfides during the synthetic sequence.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "depth")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No thioacetate protection found
        else:
            if self.condition_type == "bool":
                return 10  # Thioacetate protection is present
            else:
                # Earlier use of protection is generally better
                return 10 * (1 - x)
    
    def hit_condition(self, d):
        """
        Check if a reaction involves thioacetate protection/deprotection.
        
        Looks for:
        1. Protection: Formation of S-acetate from thiol (R-SH + AcX -> R-S-Ac)
        2. Deprotection: Hydrolysis of S-acetate to thiol (R-S-Ac -> R-SH)
        """
        if "metadata" not in d or "mapped_reaction_smiles" not in d["metadata"]:
            return False
        
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
        
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Define patterns
            thioacetate_pattern = Chem.MolFromSmarts("[#16:1]-[C:2](=O)[CH3]")  # S-Ac pattern
            thiol_pattern = Chem.MolFromSmarts("[#16:1][H]")  # S-H pattern
            
            # Check for thioacetate protection (thiol -> thioacetate)
            reactant_has_thiol = any(mol.HasSubstructMatch(thiol_pattern) for mol in reactant_mols)
            product_has_thioacetate = any(mol.HasSubstructMatch(thioacetate_pattern) for mol in product_mols)
            
            if reactant_has_thiol and product_has_thioacetate:
                return True
            
            # Check for thioacetate deprotection (thioacetate -> thiol)
            reactant_has_thioacetate = any(mol.HasSubstructMatch(thioacetate_pattern) for mol in reactant_mols)
            product_has_thiol = any(mol.HasSubstructMatch(thiol_pattern) for mol in product_mols)
            
            if reactant_has_thioacetate and product_has_thiol:
                return True
            
            # Additional check for presence of thioacetate intermediates
            all_mols = reactant_mols + product_mols
            has_thioacetate = any(mol.HasSubstructMatch(thioacetate_pattern) for mol in all_mols)
            
            return has_thioacetate
            
        except Exception:
            return False
