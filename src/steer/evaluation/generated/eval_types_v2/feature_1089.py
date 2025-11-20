"""Generated evaluation code for: Late stage Negishi coupling for fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNegishiCoupling(BaseScoring):
    """
    Evaluates whether a Negishi coupling reaction occurs in the late stage of synthesis.
    Late stage is defined as occurring after the stage_threshold fraction of the route depth.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Negishi coupling doesn't occur
        
        # x represents the depth fraction where Negishi coupling occurs
        # We want late stage (x > stage_threshold), so reward higher x values
        if x >= self.stage_threshold:
            # Scale from stage_threshold to 1.0 → score 6 to 10
            normalized = (x - self.stage_threshold) / (1.0 - self.stage_threshold)
            return 6 + 4 * normalized
        else:
            # Scale from 0 to stage_threshold → score 0 to 6
            return 6 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Detects Negishi coupling by looking for:
        1. Zinc-containing reagent in reactants
        2. C-C bond formation pattern
        3. Palladium catalyst presence (optional but common)
        """
        metadata = d.get("metadata", {})
        
        # Check for mapped reaction SMILES
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for zinc-containing reagent
            has_zinc = any(self._contains_zinc(mol) for mol in reactants)
            
            # Check for C-C bond formation pattern typical of Negishi coupling
            has_cc_formation = self._has_negishi_cc_pattern(reactants, products)
            
            return has_zinc and has_cc_formation
            
        except Exception:
            return False
    
    def _contains_zinc(self, mol) -> bool:
        """Check if molecule contains zinc atom"""
        if mol is None:
            return False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'Zn':
                return True
        return False
    
    def _has_negishi_cc_pattern(self, reactants, products) -> bool:
        """
        Check for C-C bond formation pattern consistent with Negishi coupling.
        Look for organohalide + organozinc → coupled product pattern.
        """
        # Look for halide-containing reactant (R-X where X = Cl, Br, I)
        halide_patterns = [
            Chem.MolFromSmarts("[C,c]-Cl"),
            Chem.MolFromSmarts("[C,c]-Br"), 
            Chem.MolFromSmarts("[C,c]-I")
        ]
        
        has_halide = False
        for reactant in reactants:
            if reactant and any(reactant.HasSubstructMatch(pattern) for pattern in halide_patterns if pattern):
                has_halide = True
                break
        
        # Look for organozinc pattern (C-Zn)
        organozinc_pattern = Chem.MolFromSmarts("[C,c]-Zn")
        has_organozinc = False
        if organozinc_pattern:
            for reactant in reactants:
                if reactant and reactant.HasSubstructMatch(organozinc_pattern):
                    has_organozinc = True
                    break
        
        # Simple heuristic: if we have both halide and organozinc reactants,
        # and the product has more C-C bonds than individual reactants,
        # it's likely a Negishi coupling
        if has_halide and has_organozinc:
            return True
            
        return False
