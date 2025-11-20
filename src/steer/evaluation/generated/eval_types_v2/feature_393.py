"""Generated evaluation code for: Trityl protecting group strategy for piperazine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TritylPiperazineProtection(BaseScoring):
    """
    Evaluates trityl protecting group strategy for piperazine synthesis.
    Checks if trityl protection is used during piperazine formation and 
    removed in the final deprotection step.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No trityl protection found
        if self.condition_type == "bool":
            return 1  # Found trityl protection strategy
        else:
            # Earlier protection (higher depth fraction) is better
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves trityl protection/deprotection 
        in the context of piperazine synthesis
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Check for trityl protection reaction
            if self._is_trityl_protection(reactants, products):
                return True
                
            # Check for trityl deprotection reaction
            if self._is_trityl_deprotection(reactants, products):
                return True
                
        except Exception:
            return False
            
        return False
    
    def _is_trityl_protection(self, reactants: str, products: str) -> bool:
        """Check if reaction is trityl protection of amine"""
        try:
            # Trityl chloride pattern
            trityl_chloride_pattern = "c1ccc(cc1)C(c2ccccc2)(c3ccccc3)Cl"
            trityl_chloride_smarts = Chem.MolFromSmarts("[CH]([c]1[c][c][c][c][c]1)([c]2[c][c][c][c][c]2)[Cl]")
            
            # Free amine pattern (primary or secondary)
            free_amine_smarts = Chem.MolFromSmarts("[NH2,NH1]")
            
            # Protected amine pattern (trityl-protected)
            trityl_protected_smarts = Chem.MolFromSmarts("[NH1][CH]([c]1[c][c][c][c][c]1)([c]2[c][c][c][c][c]2)")
            
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check if reactants contain trityl chloride and free amine
            has_trityl_chloride = any(mol.HasSubstructMatch(trityl_chloride_smarts) for mol in reactant_mols)
            has_free_amine = any(mol.HasSubstructMatch(free_amine_smarts) for mol in reactant_mols)
            
            # Check if products contain trityl-protected amine
            has_protected_amine = any(mol.HasSubstructMatch(trityl_protected_smarts) for mol in product_mols)
            
            return has_trityl_chloride and has_free_amine and has_protected_amine
            
        except Exception:
            return False
    
    def _is_trityl_deprotection(self, reactants: str, products: str) -> bool:
        """Check if reaction is trityl deprotection to reveal amine"""
        try:
            # Trityl-protected amine pattern
            trityl_protected_smarts = Chem.MolFromSmarts("[NH1][CH]([c]1[c][c][c][c][c]1)([c]2[c][c][c][c][c]2)")
            
            # Free amine pattern
            free_amine_smarts = Chem.MolFromSmarts("[NH2,NH1]")
            
            # Piperazine pattern to ensure context
            piperazine_smarts = Chem.MolFromSmarts("N1CCNCC1")
            
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Check if reactants contain trityl-protected amine
            has_protected_amine = any(mol.HasSubstructMatch(trityl_protected_smarts) for mol in reactant_mols)
            
            # Check if products contain free amine and piperazine
            has_free_amine = any(mol.HasSubstructMatch(free_amine_smarts) for mol in product_mols)
            has_piperazine = any(mol.HasSubstructMatch(piperazine_smarts) for mol in product_mols)
            
            return has_protected_amine and has_free_amine and has_piperazine
            
        except Exception:
            return False
