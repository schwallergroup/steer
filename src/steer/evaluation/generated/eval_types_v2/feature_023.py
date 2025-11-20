"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates synthesis routes for convergent strategy using Suzuki coupling.
    Checks if two main fragments are joined via palladium-catalyzed Suzuki cross-coupling
    at the specified timing in the route.
    """
    
    def __init__(self, config: Dict):
        self.fragments = config["parameters"]["fragments"]
        self.timing = config["parameters"]["timing"]  # "early", "middle", "late"
        
        # Suzuki coupling SMARTS patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)C(C)(C)O1")
        self.aryl_halide_pattern = Chem.MolFromSmarts("[c,C]-[Br,I]")
        
        # Target depth based on timing preference
        self.timing_map = {
            "early": 0.2,   # First 20% of route
            "middle": 0.5,  # Middle 50% of route  
            "late": 0.8     # Last 20% of route
        }
        self.target_depth = self.timing_map.get(self.timing, 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        # Score based on how close the reaction occurs to target timing
        depth_score = 1 - abs(x - self.target_depth)
        return max(0, depth_score * 10)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling between appropriate fragments"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or len(reactants) != self.fragments:
                return False
            
            # Check if we have the required Suzuki coupling partners
            has_boronic_component = False
            has_halide_component = False
            
            for reactant in reactants:
                if not reactant:
                    continue
                    
                # Check for boronic acid or ester
                if (reactant.HasSubstructMatch(self.boronic_acid_pattern) or 
                    reactant.HasSubstructMatch(self.boronic_ester_pattern)):
                    has_boronic_component = True
                
                # Check for aryl/vinyl halide
                if reactant.HasSubstructMatch(self.aryl_halide_pattern):
                    has_halide_component = True
            
            # Verify this is a genuine coupling (C-C bond formation)
            if has_boronic_component and has_halide_component:
                return self._verify_cc_bond_formation(product, reactants)
                
        except Exception:
            return False
        
        return False
    
    def _verify_cc_bond_formation(self, product, reactants) -> bool:
        """Verify that a new C-C bond is formed in the product"""
        try:
            # Count heavy atoms in reactants vs product
            reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactants if mol)
            product_heavy_atoms = product.GetNumHeavyAtoms()
            
            # In Suzuki coupling, we typically lose B(OH)2 or similar and halide
            # So product should have fewer heavy atoms than sum of reactants
            if product_heavy_atoms < reactant_heavy_atoms:
                return True
                
        except Exception:
            pass
        
        return False
