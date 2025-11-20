"""Generated evaluation code for: Azide-mediated ring formation strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideNitreneCyclization(BaseScoring):
    """
    Evaluates synthesis routes for azide-mediated ring formation via nitrene cyclization.
    
    This scoring function identifies reactions where aryl azides undergo decomposition
    to generate nitrene intermediates that participate in intramolecular ring closure.
    The score favors routes where this cyclization occurs earlier (lower depth).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        if self.condition_type == "bool":
            return 1  # Condition met
        else:
            # Earlier cyclization is better
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents an azide-mediated nitrene cyclization"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        reactants_smiles, products_smiles = mapped_rxn.split(">>")
        
        try:
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for azide functional group in reactants
            azide_pattern = Chem.MolFromSmarts("[N-]=[N+]=[N]")  # Azide group
            aryl_azide_pattern = Chem.MolFromSmarts("c[N-]=[N+]=[N]")  # Aryl azide
            
            has_azide = False
            azide_reactant = None
            
            for reactant in reactants:
                if reactant.HasSubstructMatch(azide_pattern):
                    has_azide = True
                    azide_reactant = reactant
                    break
            
            if not has_azide or not azide_reactant:
                return False
            
            # Check for ring formation in products
            reactant_ring_count = len(Chem.GetSymmSSSR(azide_reactant))
            
            for product in products:
                product_ring_count = len(Chem.GetSymmSSSR(product))
                
                # Ring formation occurred
                if product_ring_count > reactant_ring_count:
                    # Check if azide is consumed (should be absent in products)
                    if not product.HasSubstructMatch(azide_pattern):
                        # Additional check: look for nitrogen incorporation in new ring
                        nitrogen_in_ring_pattern = Chem.MolFromSmarts("[nR]")  # Aromatic N in ring
                        aliphatic_n_in_ring_pattern = Chem.MolFromSmarts("[NR]")  # Aliphatic N in ring
                        
                        if (product.HasSubstructMatch(nitrogen_in_ring_pattern) or 
                            product.HasSubstructMatch(aliphatic_n_in_ring_pattern)):
                            return True
            
            return False
            
        except Exception:
            return False
