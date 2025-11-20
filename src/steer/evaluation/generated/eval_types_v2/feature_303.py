"""Generated evaluation code for: Late stage sulfonamide formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSulfonamideFormation(BaseScoring):
    """
    Evaluates whether sulfonamide formation occurs at late stage in synthesis.
    
    Detects sulfonamide bond formation reactions and scores based on timing.
    Late-stage formation (closer to final product) receives higher scores.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.8)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sulfonamide formation doesn't occur
        
        # Score based on how late the reaction occurs
        # x is depth fraction (0 = root, 1 = leaves)
        if x >= self.stage_threshold:
            return 10 * (x - self.stage_threshold) / (1 - self.stage_threshold)
        else:
            # Penalize early-stage sulfonamide formation
            return 2 * x / self.stage_threshold
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents sulfonamide formation"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            if not prod_mol:
                return False
                
            # Check for sulfonamide formation pattern
            return self._is_sulfonamide_formation(products, reactants)
            
        except Exception:
            return False
            
    def _is_sulfonamide_formation(self, products: str, reactants: str) -> bool:
        """Detect sulfonamide bond formation"""
        try:
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) 
                           for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not reactant_mols:
                return False
                
            # Sulfonamide SMARTS pattern: S(=O)(=O)-N
            sulfonamide_pattern = Chem.MolFromSmarts("[#16](=[#8])(=[#8])-[#7]")
            if not sulfonamide_pattern:
                return False
                
            # Check if product contains sulfonamide
            has_sulfonamide_product = prod_mol.HasSubstructMatch(sulfonamide_pattern)
            if not has_sulfonamide_product:
                return False
                
            # Check if sulfonamide is newly formed (not present in all reactants)
            # Look for sulfonyl chloride or sulfonate ester reactants
            sulfonyl_chloride = Chem.MolFromSmarts("[#16](=[#8])(=[#8])-[#17]")
            sulfonate_ester = Chem.MolFromSmarts("[#16](=[#8])(=[#8])-[#8]-[#6]")
            pentafluorophenyl_sulfonate = Chem.MolFromSmarts("[#16](=[#8])(=[#8])-[#8]-c1c(F)c(F)c(F)c(F)c1F")
            
            # Check for amine reactant
            amine_pattern = Chem.MolFromSmarts("[#7;!$(N(=O)~O);!$(N=*);!$([N-]);!$(N#*)]")
            
            has_sulfonyl_reactant = False
            has_amine_reactant = False
            
            for reactant in reactant_mols:
                if (reactant.HasSubstructMatch(sulfonyl_chloride) or 
                    reactant.HasSubstructMatch(sulfonate_ester) or
                    reactant.HasSubstructMatch(pentafluorophenyl_sulfonate)):
                    has_sulfonyl_reactant = True
                    
                if reactant.HasSubstructMatch(amine_pattern):
                    has_amine_reactant = True
                    
            return has_sulfonyl_reactant and has_amine_reactant
            
        except Exception:
            return False
