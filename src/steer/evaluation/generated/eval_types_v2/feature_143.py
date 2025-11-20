"""Generated evaluation code for: Buchwald-Hartwig cross-coupling for fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BuchwaldHartwigCoupling(BaseScoring):
    """
    Evaluates synthesis routes for Buchwald-Hartwig cross-coupling reactions.
    Detects palladium-catalyzed C-N bond formation between aryl halides and amines,
    favoring early-stage convergent assembly of major fragments.
    """
    
    def __init__(self, config: Dict):
        self.min_fragments = config.get("fragments", 2)
        self.condition_type = config.get("target_depth", {}).get("type", "numerical")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Buchwald-Hartwig coupling not found
        
        if self.condition_type == "bool":
            return 1  # Reaction is present
        else:
            # Favor early-stage coupling (lower depth values)
            if x <= self.target_depth:
                return 1.0
            else:
                # Penalize late-stage coupling
                return max(0, 1.0 - (x - self.target_depth) * 2)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction represents Buchwald-Hartwig C-N coupling
        by detecting C(aryl)-N bond formation between fragments.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            # Need at least 2 reactants for coupling
            if len(reactants_smiles) < self.min_fragments:
                return False
                
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product or len(reactants) < self.min_fragments:
                return False
                
            return self._detect_buchwald_hartwig_coupling(product, reactants)
            
        except Exception:
            return False
    
    def _detect_buchwald_hartwig_coupling(self, product, reactants) -> bool:
        """
        Detects Buchwald-Hartwig coupling by checking for:
        1. C(aryl)-N bond formation
        2. Aryl halide reactant
        3. Amine/amide reactant
        """
        # Check for aryl-N bond in product
        aryl_n_pattern = Chem.MolFromSmarts("[cH,c]([#6])([#6])-[NX3]")
        if not product.HasSubstructMatch(aryl_n_pattern):
            return False
        
        # Check reactants for typical Buchwald-Hartwig substrates
        has_aryl_halide = False
        has_amine = False
        
        # Aryl halide patterns (Br, I, Cl on aromatic carbon)
        aryl_halide_patterns = [
            Chem.MolFromSmarts("c-Br"),
            Chem.MolFromSmarts("c-I"), 
            Chem.MolFromSmarts("c-Cl")
        ]
        
        # Amine patterns (primary, secondary amines, anilines)
        amine_patterns = [
            Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),  # Primary/secondary amines
            Chem.MolFromSmarts("c-[NX3;H2,H1]"),         # Anilines
            Chem.MolFromSmarts("[NX3;H1](C)C")           # Secondary amines
        ]
        
        for reactant in reactants:
            # Check for aryl halide
            if not has_aryl_halide:
                for pattern in aryl_halide_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_aryl_halide = True
                        break
            
            # Check for amine
            if not has_amine:
                for pattern in amine_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_amine = True
                        break
        
        return has_aryl_halide and has_amine
