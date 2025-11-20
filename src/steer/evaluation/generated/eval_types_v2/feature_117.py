"""Generated evaluation code for: Late stage seven-membered lactam ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSevenMemberedLactamFormation(BaseScoring):
    """
    Detects late-stage seven-membered lactam ring formation via intramolecular cyclization.
    Scores routes based on how late in the synthesis the lactam ring is formed.
    """
    
    def __init__(self, config: Dict):
        self.ring_size = config.get("ring_size", 7)
        self.timing_preference = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Lactam formation doesn't happen
        else:
            # Late-stage formation is preferred (higher score for later timing)
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a seven-membered lactam ring"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn = rxn_smiles.split(">>")
            
            if len(rxn) != 2:
                return False
                
            reactants = rxn[0]
            products = rxn[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
                
            # Check if lactam ring is formed (present in products but not reactants)
            lactam_formed = self._has_seven_membered_lactam_formation(reactant_mols, product_mols)
            
            # Verify it's intramolecular (single reactant to single product with lactam)
            if lactam_formed and len(reactant_mols) == 1 and len(product_mols) == 1:
                return self._is_intramolecular_cyclization(reactant_mols[0], product_mols[0])
                
            return False
            
        except Exception:
            return False
            
    def _has_seven_membered_lactam_formation(self, reactants, products) -> bool:
        """Check if seven-membered lactam ring is formed in this reaction"""
        # Seven-membered lactam SMARTS pattern (azepan-2-one core)
        lactam_pattern = Chem.MolFromSmarts("[#6]1[#6][#6][#6][#6][#7][#6](=[#8])1")
        
        if not lactam_pattern:
            return False
            
        # Check if lactam is present in products
        lactam_in_products = any(mol.HasSubstructMatch(lactam_pattern) for mol in products)
        
        # Check if lactam is absent in reactants  
        lactam_in_reactants = any(mol.HasSubstructMatch(lactam_pattern) for mol in reactants)
        
        return lactam_in_products and not lactam_in_reactants
        
    def _is_intramolecular_cyclization(self, reactant, product) -> bool:
        """Verify this is an intramolecular cyclization by checking atom count consistency"""
        try:
            # For intramolecular cyclization, atom counts should be similar
            # (might differ slightly due to elimination of small molecules like H2O)
            reactant_heavy_atoms = reactant.GetNumHeavyAtoms()
            product_heavy_atoms = product.GetNumHeavyAtoms()
            
            # Allow for loss of small molecules (up to 3 heavy atoms, e.g., H2O, NH3)
            atom_diff = reactant_heavy_atoms - product_heavy_atoms
            
            return 0 <= atom_diff <= 3
            
        except Exception:
            return False
