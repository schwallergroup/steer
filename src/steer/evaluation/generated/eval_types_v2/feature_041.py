"""Generated evaluation code for: Early stage Suzuki cross-coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStageSuzuki(BaseScoring):
    """
    Evaluates whether Suzuki cross-coupling for biaryl formation occurs early in the synthesis route.
    Returns higher scores when the Suzuki reaction happens before the stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Early stage is better - invert the depth fraction
            if x <= self.stage_threshold:
                return 10  # Perfect score for early stage
            else:
                # Linearly decrease score for later stages
                return max(0, 10 * (1 - (x - self.stage_threshold) / (1 - self.stage_threshold)))
    
    def hit_condition(self, d) -> bool:
        """
        Detects Suzuki cross-coupling reactions that form biaryl bonds.
        Looks for characteristic patterns: organoboron + aryl halide -> biaryl
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0].split(".")
        products = rxn_parts[1].split(".")
        
        # Check for organoboron reactant (boronic acid, boronate ester, etc.)
        has_boron = False
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    # Look for boron atoms
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() == 'B':
                            has_boron = True
                            break
            except:
                continue
        
        # Check for aryl halide reactant
        has_aryl_halide = False
        halide_patterns = [
            "[cH0:1][Cl,Br,I]",  # Aryl chloride, bromide, or iodide
            "[c:1][Cl,Br,I]"     # Alternative pattern
        ]
        
        for reactant_smiles in reactants:
            try:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is not None:
                    for pattern in halide_patterns:
                        patt = Chem.MolFromSmarts(pattern)
                        if patt is not None and mol.HasSubstructMatch(patt):
                            has_aryl_halide = True
                            break
                    if has_aryl_halide:
                        break
            except:
                continue
        
        # Check for biaryl formation in products
        has_biaryl_product = False
        biaryl_patterns = [
            "c1ccccc1-c2ccccc2",  # Simple biphenyl
            "[c:1][c:2]",         # General aryl-aryl bond
            "c-c"                 # Aromatic carbon-carbon bond
        ]
        
        for product_smiles in products:
            try:
                mol = Chem.MolFromSmiles(product_smiles)
                if mol is not None:
                    # Count aromatic rings
                    ring_info = mol.GetRingInfo()
                    aromatic_rings = 0
                    for ring in ring_info.AtomRings():
                        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                            aromatic_rings += 1
                    
                    # If multiple aromatic rings, likely biaryl formation
                    if aromatic_rings >= 2:
                        has_biaryl_product = True
                        break
            except:
                continue
        
        # Suzuki coupling requires: organoboron + aryl halide -> biaryl
        return has_boron and has_aryl_halide and has_biaryl_product
