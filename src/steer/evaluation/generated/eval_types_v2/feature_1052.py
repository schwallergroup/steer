"""Generated evaluation code for: Key bicyclic ring formation via cycloaddition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CycloadditionRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for key bicyclic ring formation via [3+2] cycloaddition.
    Detects formation of 3-azabicyclo[3.1.0]hexane cores through azomethine ylide 
    and dienophile cycloaddition reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cycloaddition doesn't occur
        else:
            # Earlier cycloaddition is generally better for key ring formation
            if self.condition_type == "bool":
                return 1  # Condition met
            else:
                return max(0, 1 - x)  # Earlier = higher score
    
    def hit_condition(self, d):
        """
        Detects [3+2] cycloaddition reactions that form bicyclic ring systems.
        Checks for characteristic pattern of two reactants combining to form 
        a bicyclic product with nitrogen in the ring system.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or len(reactant_mols) < 2:
                return False
                
            # Check if product contains bicyclic nitrogen-containing ring system
            bicyclic_n_patterns = [
                "[N]1[CH2][CH2]C2[CH2]C12",  # 3-azabicyclo[3.1.0]hexane core
                "[N]1[CH2][CH2][CH]2[CH2][CH]12",  # Alternative bicyclic pattern
                "[N]1C[CH2]C2CC12",  # Saturated version
            ]
            
            has_bicyclic_n = any(product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                               for pattern in bicyclic_n_patterns)
            
            if not has_bicyclic_n:
                return False
                
            # Check for characteristic [3+2] cycloaddition pattern:
            # Two separate reactants combining to form ring
            product_rings = product_mol.GetRingInfo().NumRings()
            total_reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactant_mols)
            
            # Ring formation occurred (product has more rings than reactants)
            ring_formation = product_rings > total_reactant_rings
            
            # Check for nitrogen-containing reactant (azomethine ylide precursor)
            has_n_reactant = any(mol.GetNumAtoms() > 0 and 
                               any(atom.GetSymbol() == 'N' for atom in mol.GetAtoms())
                               for mol in reactant_mols)
            
            # Look for alkene/dienophile reactant
            alkene_pattern = Chem.MolFromSmarts("C=C")
            has_alkene_reactant = any(mol.HasSubstructMatch(alkene_pattern) for mol in reactant_mols)
            
            return ring_formation and has_n_reactant and has_alkene_reactant
            
        except Exception:
            return False
