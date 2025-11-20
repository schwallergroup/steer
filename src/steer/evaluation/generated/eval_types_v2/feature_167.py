"""Generated evaluation code for: Late stage Cbz protection deprotection cycling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCbzCycling(MultiRxnCondBase):
    """
    Detects late-stage Cbz (carboxybenzyl) protection-deprotection cycling.
    
    Identifies routes where Cbz protection is introduced in the penultimate step
    and then removed in the final step, indicating inefficient protecting group strategy.
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Cbz protection pattern - benzyl carbamate formation
        self.cbz_pattern = Chem.MolFromSmarts("[NH1,NH2]-C(=O)-O-CH2-c1ccccc1")
        # Free amine pattern (after deprotection)
        self.free_amine_pattern = Chem.MolFromSmarts("[NH1,NH2;!$(NC=O)]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if route shows late-stage Cbz protection followed by deprotection.
        Returns (condition_met, total_reactions_count).
        """
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        # Need at least 2 reactions for protection-deprotection cycle
        if total_reactions < 2:
            return False, total_reactions
            
        # Check if final step is Cbz deprotection
        final_rxn = reactions[-1]
        is_final_deprotection = self.is_cbz_deprotection(final_rxn)
        
        if not is_final_deprotection:
            return False, total_reactions
            
        # Check if penultimate step is Cbz protection
        penultimate_rxn = reactions[-2]
        is_penultimate_protection = self.is_cbz_protection(penultimate_rxn)
        
        condition_met = is_penultimate_protection and is_final_deprotection
        return condition_met, total_reactions
    
    def is_cbz_protection(self, rxn_smiles: str) -> bool:
        """Check if reaction introduces Cbz protection."""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(smi.strip()) 
                        for smi in parts[0].split(".") if smi.strip()]
            product = Chem.MolFromSmiles(parts[1].strip())
            
            if not all(reactants) or not product:
                return False
                
            # Product should contain Cbz group
            has_cbz_in_product = product.HasSubstructMatch(self.cbz_pattern)
            
            # Reactants should not contain Cbz group (or contain fewer)
            cbz_in_reactants = sum(1 for mol in reactants 
                                 if mol.HasSubstructMatch(self.cbz_pattern))
            cbz_in_product = len(product.GetSubstructMatches(self.cbz_pattern))
            
            return has_cbz_in_product and cbz_in_product > cbz_in_reactants
            
        except Exception:
            return False
    
    def is_cbz_deprotection(self, rxn_smiles: str) -> bool:
        """Check if reaction removes Cbz protection."""
        try:
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
                
            reactants = [Chem.MolFromSmiles(smi.strip()) 
                        for smi in parts[0].split(".") if smi.strip()]
            product = Chem.MolFromSmiles(parts[1].strip())
            
            if not all(reactants) or not product:
                return False
                
            # Check if reactant has Cbz and product has free amine
            has_cbz_in_reactant = any(mol.HasSubstructMatch(self.cbz_pattern) 
                                    for mol in reactants)
            has_free_amine_in_product = product.HasSubstructMatch(self.free_amine_pattern)
            
            # Count Cbz groups - should decrease from reactants to product
            cbz_in_reactants = sum(len(mol.GetSubstructMatches(self.cbz_pattern)) 
                                 for mol in reactants)
            cbz_in_product = len(product.GetSubstructMatches(self.cbz_pattern))
            
            return (has_cbz_in_reactant and has_free_amine_in_product and 
                   cbz_in_product < cbz_in_reactants)
            
        except Exception:
            return False
