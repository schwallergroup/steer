"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage diaryl ether formation.
    Checks if a diaryl ether bond (c-O-c) is formed late in the synthesis,
    which is typical for nucleophilic aromatic substitution reactions.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config.get("bond_smarts", "c-O-c")
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diaryl ether formation doesn't happen
        else:
            # For late-stage formation, lower depth fraction is better
            # Convert to 0-10 scale where late formation scores higher
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a diaryl ether bond"""
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
            
            # Check if product contains diaryl ether pattern
            ether_pattern = Chem.MolFromSmarts(self.bond_smarts)
            if not product_mol.HasSubstructMatch(ether_pattern):
                return False
                
            # Check if the ether bond is newly formed (not present in reactants)
            ether_in_reactants = any(
                reactant.HasSubstructMatch(ether_pattern) 
                for reactant in reactant_mols
            )
            
            # Return True if ether is in product but not in reactants (bond formation)
            return not ether_in_reactants
            
        except Exception:
            return False
