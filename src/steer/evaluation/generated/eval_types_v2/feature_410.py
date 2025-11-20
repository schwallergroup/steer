"""Generated evaluation code for: Late stage diaryl ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageDialylEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage diaryl ether formation.
    Checks if a diaryl ether bond (Ar-O-Ar) is formed in the later stages of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)  # Default to late stage
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diaryl ether formation doesn't happen
        else:
            # Late-stage formation is better, so higher depth fraction gives higher score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves diaryl ether formation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".") if r.strip()]
            
            if not product or not all(reactants):
                return False
            
            # Check if diaryl ether is formed in this step
            return self._is_diaryl_ether_formation(product, reactants)
            
        except Exception:
            return False
    
    def _is_diaryl_ether_formation(self, product, reactants) -> bool:
        """
        Check if a diaryl ether bond (Ar-O-Ar) is formed in this reaction.
        """
        # SMARTS pattern for diaryl ether: aromatic carbon - oxygen - aromatic carbon
        diaryl_ether_pattern = Chem.MolFromSmarts("[cR]-O-[cR]")
        
        if not diaryl_ether_pattern:
            return False
        
        # Check if product contains diaryl ether
        product_matches = product.GetSubstructMatches(diaryl_ether_pattern)
        if not product_matches:
            return False
        
        # Check if any reactant already contains the same diaryl ether
        # If so, it's not formation but already present
        for reactant in reactants:
            reactant_matches = reactant.GetSubstructMatches(diaryl_ether_pattern)
            
            # Compare atom mappings to see if the same diaryl ether exists in reactants
            for prod_match in product_matches:
                prod_atoms = [product.GetAtomWithIdx(idx).GetAtomMapNum() for idx in prod_match]
                
                for react_match in reactant_matches:
                    react_atoms = [reactant.GetAtomWithIdx(idx).GetAtomMapNum() for idx in react_match]
                    
                    # If the same mapped atoms form diaryl ether in both product and reactant,
                    # this is not a formation reaction
                    if set(prod_atoms) == set(react_atoms) and all(atom > 0 for atom in prod_atoms):
                        return False
        
        # Additional check: ensure we have aromatic coupling
        # Look for reaction pattern where aromatic halide/pseudohalide reacts with phenol
        phenol_pattern = Chem.MolFromSmarts("[cR]-O")
        aryl_halide_pattern = Chem.MolFromSmarts("[cR]-[F,Cl,Br,I]")
        
        has_phenol = any(reactant.HasSubstructMatch(phenol_pattern) for reactant in reactants)
        has_aryl_halide = any(reactant.HasSubstructMatch(aryl_halide_pattern) for reactant in reactants)
        
        # Return True if we have diaryl ether in product and typical coupling partners in reactants
        return has_phenol or has_aryl_halide
