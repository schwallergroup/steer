"""Generated evaluation code for: Late stage reductive amination for final assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageReductiveAmination(BaseScoring):
    """
    Checks if reductive amination occurs at a late stage (final assembly step).
    Detects the formation of C-N bonds from carbonyl compounds and amines.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")  # "late" means lower depth is better
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reductive amination doesn't occur
        else:
            # Late stage (lower depth fraction) is preferred for final assembly
            return 1 - x  # Score closer to 1 for earlier occurrence
    
    def hit_condition(self, d) -> bool:
        """
        Detect reductive amination by identifying:
        1. Formation of new C-N bond
        2. Loss of C=O (carbonyl) functionality
        3. Presence of amine reactant and carbonyl reactant
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if r]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for carbonyl reactant (aldehyde or ketone)
            carbonyl_pattern = Chem.MolFromSmarts("[CX3]=[OX1]")
            has_carbonyl_reactant = any(mol.HasSubstructMatch(carbonyl_pattern) for mol in reactant_mols)
            
            # Check for amine reactant (primary or secondary amine)
            amine_pattern = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")
            has_amine_reactant = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Check for new C-N bond formation in product
            # Look for aliphatic C-N bonds that would result from reductive amination
            cn_bond_pattern = Chem.MolFromSmarts("[CX4]-[NX3]")
            has_cn_product = product_mol.HasSubstructMatch(cn_bond_pattern)
            
            # Verify carbonyl is consumed (less carbonyl groups in product than total in reactants)
            reactant_carbonyls = sum(len(mol.GetSubstructMatches(carbonyl_pattern)) for mol in reactant_mols)
            product_carbonyls = len(product_mol.GetSubstructMatches(carbonyl_pattern))
            carbonyl_consumed = reactant_carbonyls > product_carbonyls
            
            return (has_carbonyl_reactant and has_amine_reactant and 
                   has_cn_product and carbonyl_consumed)
                   
        except Exception:
            return False
