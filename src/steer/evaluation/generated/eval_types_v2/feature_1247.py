"""Generated evaluation code for: Early tricyclic core formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TricyclicCoreFormation(BaseScoring):
    """
    Evaluates routes for early tricyclic core formation via cyclization.
    Checks if a reaction forms the specified number of rings and occurs early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.target_step = config["parameters"]["step_number"]
        self.rings_formed = config["parameters"]["rings_formed"]
        self.timing = config["parameters"]["timing"]  # "early" timing preference
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        # For early timing, reward earlier occurrence (lower depth values)
        if self.timing == "early":
            return max(0, 1 - x)  # Earlier is better, score decreases with depth
        else:
            # For other timing preferences, use step-based scoring
            return max(0, 1 - abs(x - self.target_step / 10))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves tricyclic core formation via cyclization."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles:
                return False
            
            # Split reaction into reactants and products
            parts = rxn_smiles.split(">>")
            if len(parts) != 2:
                return False
            
            reactants_smiles = parts[0]
            products_smiles = parts[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Count rings in reactants and products
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
            
            # Check if the expected number of rings were formed
            rings_formed = product_rings - reactant_rings
            
            # Additional check for intramolecular cyclization pattern
            # Look for cyclization by checking if we have fewer molecules in products
            is_cyclization = len(reactants) > len(products) or self._is_intramolecular_cyclization(reactants, products)
            
            return rings_formed >= self.rings_formed and is_cyclization
            
        except Exception:
            return False
    
    def _is_intramolecular_cyclization(self, reactants, products) -> bool:
        """Check if the reaction appears to be an intramolecular cyclization."""
        try:
            # Simple heuristic: if we have one main reactant and one main product
            # with increased ring count, it's likely intramolecular cyclization
            if len(reactants) == 1 and len(products) == 1:
                reactant_rings = reactants[0].GetRingInfo().NumRings()
                product_rings = products[0].GetRingInfo().NumRings()
                return product_rings > reactant_rings
            
            # For multi-component reactions, check if main organic molecules
            # have increased ring count (excluding small molecules like water, etc.)
            large_reactants = [mol for mol in reactants if mol.GetNumAtoms() > 5]
            large_products = [mol for mol in products if mol.GetNumAtoms() > 5]
            
            if large_reactants and large_products:
                max_reactant_rings = max(mol.GetRingInfo().NumRings() for mol in large_reactants)
                max_product_rings = max(mol.GetRingInfo().NumRings() for mol in large_products)
                return max_product_rings > max_reactant_rings
                
            return False
            
        except Exception:
            return False
