"""Generated evaluation code for: Late stage intramolecular lactam formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LactamFormationDepth(BaseScoring):
    """
    Evaluates the depth at which intramolecular lactam formation occurs in a synthesis route.
    Specifically looks for late-stage formation of lactam rings matching the given SMARTS pattern.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.formation_type = config["parameters"]["formation_type"]
        self.lactam_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Lactam formation doesn't happen
        else:
            # For late-stage timing, higher depth fraction is better (closer to 1.0)
            if self.timing == "late":
                return x * 10  # Convert to 0-10 scale, favoring late formation
            else:
                return (1 - x) * 10  # Convert to 0-10 scale, favoring early formation
    
    def hit_condition(self, d):
        """
        Checks if this reaction step involves intramolecular lactam formation.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        # Parse reactants and products
        try:
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
                
        except:
            return False
        
        # Check if lactam ring is formed (present in products but not reactants)
        lactam_in_products = any(mol.HasSubstructMatch(self.lactam_pattern) for mol in products if mol)
        lactam_in_reactants = any(mol.HasSubstructMatch(self.lactam_pattern) for mol in reactants if mol)
        
        if not lactam_in_products or lactam_in_reactants:
            return False  # No new lactam formation
        
        # Check for intramolecular formation
        if self.formation_type == "intramolecular":
            return self._is_intramolecular_cyclization(reactants, products)
        
        return True
    
    def _is_intramolecular_cyclization(self, reactants, products):
        """
        Checks if the lactam formation is intramolecular by verifying that:
        1. There are fewer product molecules than reactant molecules
        2. The molecular formula is conserved (accounting for water elimination)
        """
        # Simple heuristic: intramolecular cyclization typically reduces molecule count
        if len(products) >= len(reactants):
            return False
        
        # Check if we have a single reactant forming the lactam (most common intramolecular case)
        if len(reactants) == 1 and len(products) >= 1:
            return True
        
        # For multiple reactants, check if total heavy atom count decreases appropriately
        # (indicating cyclization with possible elimination of small molecules like H2O)
        reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactants if mol)
        product_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in products if mol)
        
        # Allow for loss of small molecules (H2O, etc.) during cyclization
        return reactant_heavy_atoms >= product_heavy_atoms
