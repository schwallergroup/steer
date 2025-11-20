"""Generated evaluation code for: Late stage halogen exchange bromide to iodide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageHalogenExchange(BaseScoring):
    """
    Evaluates synthesis routes for late-stage halogen exchange from bromide to iodide.
    Checks if C-Br bond is converted to C-I bond, preferably in final steps.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        self.target_step = config.get("reaction_step", "final")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No halogen exchange found
        
        if self.timing_preference == "late":
            # Reward later stages more (closer to 1.0 depth fraction)
            return 10 * x
        else:
            # Standard scoring - earlier is better
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction performs C-Br to C-I halogen exchange"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            product_smiles, reactant_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Get atom mappings for bromines and iodines
            product_atoms = {atom.GetAtomMapNum(): atom for atom in product.GetAtoms() if atom.GetAtomMapNum() > 0}
            
            # Find bromines in product that were iodines in reactants
            for atom_map_num, prod_atom in product_atoms.items():
                if prod_atom.GetSymbol() == 'I':  # Iodine in product
                    # Check if this mapped atom was Br in any reactant
                    for reactant in reactants:
                        for react_atom in reactant.GetAtoms():
                            if (react_atom.GetAtomMapNum() == atom_map_num and 
                                react_atom.GetSymbol() == 'Br'):
                                # Verify it's attached to carbon in both
                                if self._is_carbon_halogen_bond(prod_atom, product) and \
                                   self._is_carbon_halogen_bond(react_atom, reactant):
                                    return True
            
            return False
            
        except Exception:
            return False
    
    def _is_carbon_halogen_bond(self, halogen_atom, mol) -> bool:
        """Check if halogen atom is bonded to carbon"""
        for neighbor in halogen_atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                return True
        return False
