"""Generated evaluation code for: Late stage intramolecular cyclization for cyclopentanol formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCyclopentanolFormation(BaseScoring):
    """
    Evaluates late-stage intramolecular cyclization for cyclopentanol formation.
    Checks if a cyclopentane ring is formed via intramolecular cyclization late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_type = config["parameters"]["formation_type"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # For late-stage preference, later formation gets higher score
            if self.timing == "late":
                return 1 - x  # x is depth fraction, so 1-x rewards later stages
            else:
                return x  # Early formation would be rewarded with x
                
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a cyclopentane ring intramolecularly"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse products and reactants
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
                
            # Check if cyclopentane ring is formed (present in products but not reactants)
            ring_in_products = any(mol.HasSubstructMatch(self.ring_pattern) for mol in products)
            ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactants)
            
            if not (ring_in_products and not ring_in_reactants):
                return False
                
            # Check for intramolecular cyclization
            # This means: single reactant forms ring-containing product + possible small molecules
            if self.formation_type == "intramolecular":
                return self._is_intramolecular_cyclization(reactants, products)
            
            return True
            
        except Exception:
            return False
            
    def _is_intramolecular_cyclization(self, reactants, products):
        """Check if this is an intramolecular cyclization reaction"""
        # Find the main reactant (largest molecule without the ring)
        main_reactant = None
        for reactant in reactants:
            if not reactant.HasSubstructMatch(self.ring_pattern):
                if main_reactant is None or reactant.GetNumAtoms() > main_reactant.GetNumAtoms():
                    main_reactant = reactant
                    
        if main_reactant is None:
            return False
            
        # Find the ring-containing product
        ring_product = None
        for product in products:
            if product.HasSubstructMatch(self.ring_pattern):
                ring_product = product
                break
                
        if ring_product is None:
            return False
            
        # For intramolecular cyclization, the main reactant should have similar
        # or more atoms than the ring product (accounting for small leaving groups)
        reactant_atoms = main_reactant.GetNumAtoms()
        product_atoms = ring_product.GetNumAtoms()
        
        # Allow for loss of small molecules (up to 5 atoms difference)
        return reactant_atoms >= product_atoms and (reactant_atoms - product_atoms) <= 5
