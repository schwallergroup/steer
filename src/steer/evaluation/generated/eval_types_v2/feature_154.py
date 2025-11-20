"""Generated evaluation code for: Late piperazine ring formation via intramolecular cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageRingFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage intramolecular piperazine ring formation.
    Rewards routes where the target ring is formed in the final steps via cyclization.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late"
        self.formation_method = config["parameters"]["formation_method"]  # "intramolecular"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Later formation is better for "late" timing
            if self.timing == "late":
                return 1 - x  # Higher score for formation closer to final product
            else:
                return x  # Earlier formation preferred
                
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves intramolecular piperazine ring formation"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            # Check if target ring is present in products but not in any single reactant
            has_ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products if mol)
            has_ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants if mol)
            
            if not has_ring_in_products or has_ring_in_reactants:
                return False
                
            # For intramolecular cyclization, check that we go from one reactant to one product
            # and the reactant contains the open-chain precursor
            if self.formation_method == "intramolecular":
                return self._is_intramolecular_cyclization(reactants, products)
                
            return True
            
        except Exception:
            return False
            
    def _is_intramolecular_cyclization(self, reactants, products):
        """Check if this is an intramolecular cyclization forming the target ring"""
        # Look for a single main reactant that becomes the ring-containing product
        main_reactants = [mol for mol in reactants if mol and mol.GetNumAtoms() > 5]
        main_products = [mol for mol in products if mol and mol.HasSubstructMatch(self.ring_pattern)]
        
        if len(main_products) != 1:
            return False
            
        # Check if any reactant has the open-chain precursor pattern
        # Piperazine precursor pattern: contains two nitrogens that could cyclize
        precursor_pattern = Chem.MolFromSmarts("N-C-C-C-N")  # Open chain with two N atoms
        
        for reactant in main_reactants:
            if reactant.HasSubstructMatch(precursor_pattern):
                # Check atom count consistency (allowing for small leaving groups)
                reactant_heavy_atoms = reactant.GetNumHeavyAtoms()
                product_heavy_atoms = main_products[0].GetNumHeavyAtoms()
                
                # Allow for loss of small leaving groups (up to 3 heavy atoms)
                if reactant_heavy_atoms - product_heavy_atoms <= 3:
                    return True
                    
        return False
