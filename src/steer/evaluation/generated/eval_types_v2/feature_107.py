"""Generated evaluation code for: Three-component pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ThreeComponentPyrazoleFormation(BaseScoring):
    """
    Detects three-component pyrazole ring formation reactions in synthesis routes.
    Checks for reactions that form pyrazole rings from exactly 3 reactants.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition not met
                return 1 if x < 0 else 0
        else:
            if x < 0:
                return 0
            return abs(x - self.target_depth)
    
    def hit_condition(self, d):
        # Check if reaction has exactly 3 reactants
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Must have exactly 3 reactants
        if len(reactants) != 3:
            return False
        
        # Check if pyrazole ring is formed
        return self._pyrazole_ring_formed(reactants, products)
    
    def _pyrazole_ring_formed(self, reactants, products):
        """Check if a pyrazole ring is formed in the reaction"""
        try:
            # Pyrazole SMARTS pattern
            pyrazole_pattern = Chem.MolFromSmarts("c1nn[cH][cH]1")
            if not pyrazole_pattern:
                return False
            
            # Check if any reactant already contains pyrazole
            reactant_has_pyrazole = False
            for r_smiles in reactants:
                try:
                    reactant_mol = Chem.MolFromSmiles(r_smiles)
                    if reactant_mol and reactant_mol.HasSubstructMatch(pyrazole_pattern):
                        reactant_has_pyrazole = True
                        break
                except:
                    continue
            
            # If reactants already have pyrazole, this isn't formation
            if reactant_has_pyrazole:
                return False
            
            # Check if products contain pyrazole
            product_has_pyrazole = False
            for p_smiles in products:
                try:
                    product_mol = Chem.MolFromSmiles(p_smiles)
                    if product_mol and product_mol.HasSubstructMatch(pyrazole_pattern):
                        product_has_pyrazole = True
                        break
                except:
                    continue
            
            return product_has_pyrazole
            
        except Exception:
            return False
