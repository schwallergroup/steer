"""Generated evaluation code for: Late stage primary amine acetylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmineAcetylation(BaseScoring):
    """
    Evaluates whether a primary amine acetylation reaction occurs at a specific position from the end.
    The reaction converts a primary amine to acetamide, typically using acetyl chloride or acetic anhydride.
    """
    
    def __init__(self, config: Dict):
        self.position_from_end = config.get("position_from_end", 1)
        # SMARTS patterns for primary amine and acetamide
        self.primary_amine_pattern = "[NH2][#6]"
        self.acetamide_pattern = "[NH1]([#6])C(=O)C"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Perfect score if at exact position, penalty for being too early
            target_position = 1.0 - (1.0 / self.position_from_end) if self.position_from_end > 0 else 0.0
            penalty = abs(x - target_position) * 10
            return max(0, 10 - penalty)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents primary amine acetylation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains acetamide pattern
            acetamide_pattern_mol = Chem.MolFromSmarts(self.acetamide_pattern)
            has_acetamide = product_mol.HasSubstructMatch(acetamide_pattern_mol)
            
            if not has_acetamide:
                return False
            
            # Check if any reactant contains primary amine pattern
            primary_amine_pattern_mol = Chem.MolFromSmarts(self.primary_amine_pattern)
            has_primary_amine = any(mol.HasSubstructMatch(primary_amine_pattern_mol) for mol in reactant_mols)
            
            if not has_primary_amine:
                return False
            
            # Additional check: look for acetylating agent (acetyl chloride, acetic anhydride, etc.)
            acetyl_chloride_pattern = Chem.MolFromSmarts("C(=O)Cl")
            acetic_anhydride_pattern = Chem.MolFromSmarts("C(=O)OC(=O)C")
            
            has_acetylating_agent = any(
                mol.HasSubstructMatch(acetyl_chloride_pattern) or 
                mol.HasSubstructMatch(acetic_anhydride_pattern) 
                for mol in reactant_mols
            )
            
            return has_acetylating_agent
            
        except Exception:
            return False
