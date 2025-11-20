"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction for biaryl formation occurs at a late stage.
    Rewards routes where Suzuki coupling happens closer to the end of the synthesis.
    """
    
    def __init__(self, config: Dict):
        # Config for timing preference - late stage is better
        self.prefer_late = config.get("prefer_late", True)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            # Late-stage (higher depth fraction) is rewarded more
            return x * 10  # Convert depth fraction to 0-10 score
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki coupling reactions that form biaryl bonds.
        Looks for characteristic patterns of aryl-aryl bond formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(reactant_mols):
                return False
                
            # Check if this is a Suzuki coupling by looking for:
            # 1. Biaryl formation in product
            # 2. Presence of boronic acid/ester in reactants
            # 3. Aryl halide in reactants
            
            has_biaryl_formation = self._detect_biaryl_formation(prod_mol, reactant_mols)
            has_boronic_component = self._detect_boronic_acid_or_ester(reactant_mols)
            has_aryl_halide = self._detect_aryl_halide(reactant_mols)
            
            return has_biaryl_formation and has_boronic_component and has_aryl_halide
            
        except Exception:
            return False
    
    def _detect_biaryl_formation(self, product, reactants):
        """Check if a biaryl bond is formed in the product."""
        # Pattern for biaryl: two aromatic rings connected by single bond
        biaryl_pattern = Chem.MolFromSmarts("[c]:[c]-[c]:[c]")
        
        if not biaryl_pattern:
            return False
            
        # Product should have biaryl
        if not product.HasSubstructMatch(biaryl_pattern):
            return False
            
        # At least one reactant should lack this biaryl connection
        for reactant in reactants:
            if reactant and not reactant.HasSubstructMatch(biaryl_pattern):
                return True
                
        return False
    
    def _detect_boronic_acid_or_ester(self, reactants):
        """Detect boronic acid or boronic ester in reactants."""
        # Boronic acid pattern: R-B(OH)2
        boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-O)-O")
        # Boronic ester patterns: pinacol ester, etc.
        boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1-O[C][C]O-1")
        
        if not boronic_acid_pattern or not boronic_ester_pattern:
            return False
            
        for reactant in reactants:
            if reactant and (reactant.HasSubstructMatch(boronic_acid_pattern) or 
                           reactant.HasSubstructMatch(boronic_ester_pattern)):
                return True
                
        return False
    
    def _detect_aryl_halide(self, reactants):
        """Detect aryl halide in reactants."""
        # Aromatic carbon bonded to halogen
        aryl_halide_patterns = [
            Chem.MolFromSmarts("[c]-Cl"),
            Chem.MolFromSmarts("[c]-Br"), 
            Chem.MolFromSmarts("[c]-I"),
            Chem.MolFromSmarts("[c]-F")
        ]
        
        for pattern in aryl_halide_patterns:
            if pattern:
                for reactant in reactants:
                    if reactant and reactant.HasSubstructMatch(pattern):
                        return True
                        
        return False
