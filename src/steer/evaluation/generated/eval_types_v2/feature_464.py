"""Generated evaluation code for: Late stage Suzuki coupling for biphenyl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs at late stage for biphenyl formation.
    
    Detects Suzuki coupling reactions that form C-C bonds between two aromatic rings,
    with preference for reactions occurring later in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Prefer late stage
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Later stage reactions get higher scores.
        """
        if x < 0:
            return 0  # Suzuki coupling not found
        
        if self.condition_type == "bool":
            return 10 if x >= self.target_depth else 0
        else:
            # Linear scoring favoring late stage (higher depth fraction)
            return min(10, x * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a Suzuki coupling forming biphenyl.
        """
        metadata = d.get("metadata", {})
        
        # Check if reaction SMILES is available
        if "mapped_reaction_smiles" not in metadata:
            return False
        
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
        
        try:
            # Parse product and reactants
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactant_smiles = rxn_parts[1].split(".")
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles if r]
            
            if not product or not reactants:
                return False
            
            # Check for Suzuki coupling characteristics:
            # 1. At least two reactants (aryl halide + boronic acid/ester)
            # 2. Formation of new C-C bond between aromatic rings
            
            return (self._has_suzuki_reactants(reactants) and 
                    self._forms_biphenyl_bond(product, reactants))
        
        except Exception:
            return False
    
    def _has_suzuki_reactants(self, reactants) -> bool:
        """Check for typical Suzuki coupling reactants."""
        has_aryl_halide = False
        has_boron_species = False
        
        # Aryl halide pattern (aromatic ring with halogen)
        aryl_halide_pattern = Chem.MolFromSmarts("[cH0,c:1]-[Br,I,Cl]")
        
        # Boronic acid/ester patterns
        boronic_acid_pattern = Chem.MolFromSmarts("[c:1]-B(O)(O)")
        boronic_ester_pattern = Chem.MolFromSmarts("[c:1]-B1OCCCO1")  # Pinacol ester
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(aryl_halide_pattern):
                has_aryl_halide = True
            
            if (reactant.HasSubstructMatch(boronic_acid_pattern) or 
                reactant.HasSubstructMatch(boronic_ester_pattern)):
                has_boron_species = True
        
        return has_aryl_halide and has_boron_species
    
    def _forms_biphenyl_bond(self, product, reactants) -> bool:
        """
        Check if the reaction forms a new C-C bond between aromatic rings
        by comparing aromatic ring connectivity in product vs reactants.
        """
        # Count aromatic rings connected by single bonds in product
        biphenyl_pattern = Chem.MolFromSmarts("c1ccccc1-c2ccccc2")
        product_biphenyls = len(product.GetSubstructMatches(biphenyl_pattern))
        
        # Count aromatic rings connected by single bonds in all reactants
        reactant_biphenyls = 0
        for reactant in reactants:
            reactant_biphenyls += len(reactant.GetSubstructMatches(biphenyl_pattern))
        
        # New biphenyl bond formed if product has more than reactants
        return product_biphenyls > reactant_biphenyls
