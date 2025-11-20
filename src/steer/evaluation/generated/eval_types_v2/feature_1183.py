"""Generated evaluation code for: Late stage thiol protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiolProtection(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiol protection with trityl group.
    Checks if a thiol group is protected with a trityl group at a specific late stage.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config.get("protecting_group", "trityl")
        self.step_position = config.get("step_position", 1)
        self.timing = config.get("timing", "late")
        
        # Define SMARTS patterns
        self.free_thiol_pattern = "[SH1]"  # Free thiol group
        self.trityl_protected_thiol_pattern = "[S;X2]-[C]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)[c]3[cH][cH][cH][cH][cH]3"  # Trityl-protected thiol
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection doesn't happen
        
        # For late-stage protection, we want it close to the end (depth near 1.0)
        if self.timing == "late":
            if x > 0.8:  # Very late stage
                return 10
            elif x > 0.6:  # Moderately late
                return 7
            elif x > 0.4:  # Mid-stage
                return 4
            else:  # Early stage
                return 1
        else:
            # General case - prefer later stages
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves thiol protection with trityl group.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not reactant_mols:
                return False
            
            # Check if product has trityl-protected thiol
            trityl_pattern = Chem.MolFromSmarts(self.trityl_protected_thiol_pattern)
            has_protected_product = prod_mol.HasSubstructMatch(trityl_pattern)
            
            # Check if any reactant has free thiol
            thiol_pattern = Chem.MolFromSmarts(self.free_thiol_pattern)
            has_free_thiol_reactant = any(mol.HasSubstructMatch(thiol_pattern) for mol in reactant_mols)
            
            # Check if trityl reagent is present in reactants
            trityl_reagent_pattern = Chem.MolFromSmarts("[C]([c]1[cH][cH][cH][cH][cH]1)([c]2[cH][cH][cH][cH][cH]2)([c]3[cH][cH][cH][cH][cH]3)[Cl,Br,I]")  # Trityl halide
            has_trityl_reagent = any(mol.HasSubstructMatch(trityl_reagent_pattern) for mol in reactant_mols)
            
            # Protection reaction: free thiol + trityl reagent -> protected thiol
            return has_protected_product and has_free_thiol_reactant and has_trityl_reagent
            
        except Exception:
            return False
