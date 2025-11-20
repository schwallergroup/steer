"""Generated evaluation code for: Late stage N-alkynylation on imidazole"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNAlkynylation(BaseScoring):
    """
    Evaluates routes for late-stage N-alkynylation reactions on imidazole substrates.
    Rewards routes where N-alkynylation occurs as late as possible in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = config.get("substrate_pattern", "[nH]1c[nH]cc1")  # imidazole pattern
        self.substrate_mol = Chem.MolFromSmarts(self.substrate_pattern)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-alkynylation doesn't happen
        else:
            # Late-stage is better - higher score for reactions closer to end (lower depth fraction)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents N-alkynylation on an imidazole substrate.
        """
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
            
            # Check if product contains imidazole with alkyne
            has_imidazole_alkyne = self._has_imidazole_alkyne(product_mol)
            if not has_imidazole_alkyne:
                return False
            
            # Check if any reactant has imidazole without alkyne attached to N
            has_imidazole_reactant = any(self._has_free_imidazole(mol) for mol in reactant_mols)
            if not has_imidazole_reactant:
                return False
                
            # Check if any reactant has alkyne functionality
            has_alkyne_reactant = any(self._has_alkyne(mol) for mol in reactant_mols)
            
            return has_alkyne_reactant
            
        except Exception:
            return False
    
    def _has_imidazole_alkyne(self, mol) -> bool:
        """Check if molecule has imidazole with N-alkyne substitution."""
        # Pattern for imidazole with N-alkyne: imidazole nitrogen connected to alkyne
        alkyne_imidazole_pattern = Chem.MolFromSmarts("[n]1c[nH]cc1-C#C")
        alkyne_imidazole_pattern2 = Chem.MolFromSmarts("[nH]1cn(-C#C)cc1")
        
        return (mol.HasSubstructMatch(alkyne_imidazole_pattern) or 
                mol.HasSubstructMatch(alkyne_imidazole_pattern2))
    
    def _has_free_imidazole(self, mol) -> bool:
        """Check if molecule has imidazole without alkyne attached."""
        if not mol.HasSubstructMatch(self.substrate_mol):
            return False
        
        # Make sure it doesn't already have N-alkyne
        return not self._has_imidazole_alkyne(mol)
    
    def _has_alkyne(self, mol) -> bool:
        """Check if molecule contains alkyne functionality."""
        alkyne_pattern = Chem.MolFromSmarts("C#C")
        return mol.HasSubstructMatch(alkyne_pattern)
