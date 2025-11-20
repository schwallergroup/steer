"""Generated evaluation code for: Early intramolecular cyclization to form imide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyImideCyclization(BaseScoring):
    """
    Evaluates whether intramolecular cyclization to form an imide ring occurs early in the synthesis route.
    
    An imide is defined as a compound containing a nitrogen atom bonded to two carbonyl carbons.
    Early timing is rewarded with higher scores.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fractional")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Early stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Cyclization doesn't happen
        else:
            # Early cyclization (lower x) gets higher score
            return max(0, 1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves intramolecular cyclization to form an imide.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Define imide pattern: nitrogen bonded to two carbonyls
            imide_pattern = Chem.MolFromSmarts("[#6](=[#8])-[#7]-[#6](=[#8])")
            
            # Check if products have imide but reactants don't (ring formation)
            products_have_imide = any(mol.HasSubstructMatch(imide_pattern) for mol in products)
            reactants_have_imide = any(mol.HasSubstructMatch(imide_pattern) for mol in reactants)
            
            if not (products_have_imide and not reactants_have_imide):
                return False
            
            # Check for intramolecularity: 
            # Should have fewer molecules in products than reactants (cyclization reduces molecule count)
            # and the imide-containing product should have same heavy atom count as major reactant
            if len(products) >= len(reactants):
                return False
            
            # Find the imide-containing product
            imide_product = None
            for prod in products:
                if prod.HasSubstructMatch(imide_pattern):
                    imide_product = prod
                    break
            
            if imide_product is None:
                return False
            
            # Check if this could be intramolecular by comparing heavy atom counts
            prod_heavy_atoms = imide_product.GetNumHeavyAtoms()
            
            # Find a reactant with similar heavy atom count (allowing for loss of small molecules)
            for react in reactants:
                react_heavy_atoms = react.GetNumHeavyAtoms()
                # Allow for loss of water or small molecules in cyclization
                if abs(prod_heavy_atoms - react_heavy_atoms) <= 2:
                    return True
            
            return False
            
        except Exception:
            return False
