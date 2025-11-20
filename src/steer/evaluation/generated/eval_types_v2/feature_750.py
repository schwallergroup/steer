"""Generated evaluation code for: Late stage amide coupling for fragment connection"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Detects late-stage amide coupling reactions used for fragment assembly.
    Looks for formation of amide bonds (C(=O)N) where both fragments contain
    significant structural complexity, indicating fragment connection rather
    than simple functionalization.
    """
    
    def __init__(self, config: Dict):
        self.min_fragment_complexity = config.get("min_fragment_complexity", 8)
        self.target_stage = config.get("target_stage", "late")  # "late", "early", "any"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Amide coupling doesn't happen
        
        if self.target_stage == "late":
            return 1 - x  # Later stage is better (higher score for lower depth fraction)
        elif self.target_stage == "early":
            return x  # Earlier stage is better
        else:  # "any"
            return 1 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an amide coupling between significant fragments."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if amide bond is formed
            if not self._amide_bond_formed(product, reactants):
                return False
            
            # Check if both fragments have sufficient complexity
            if not self._fragments_are_complex(reactants):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _amide_bond_formed(self, product, reactants) -> bool:
        """Check if an amide bond is formed in this reaction."""
        # Amide pattern: C(=O)N
        amide_pattern = Chem.MolFromSmarts("[C](=[O])[N]")
        
        # Count amides in product vs reactants
        product_amides = len(product.GetSubstructMatches(amide_pattern))
        reactant_amides = sum(len(r.GetSubstructMatches(amide_pattern)) for r in reactants)
        
        # Check if new amide bond formed
        return product_amides > reactant_amides
    
    def _fragments_are_complex(self, reactants) -> bool:
        """Check if reactants represent significant fragments (not simple coupling reagents)."""
        # Filter out small coupling reagents/activators
        significant_fragments = []
        
        for reactant in reactants:
            # Skip small molecules likely to be reagents
            if reactant.GetNumHeavyAtoms() < 4:
                continue
                
            # Skip common coupling reagents by checking for specific patterns
            if self._is_coupling_reagent(reactant):
                continue
                
            significant_fragments.append(reactant)
        
        # Need at least 2 significant fragments, both with sufficient complexity
        if len(significant_fragments) < 2:
            return False
            
        return all(frag.GetNumHeavyAtoms() >= self.min_fragment_complexity 
                  for frag in significant_fragments[:2])
    
    def _is_coupling_reagent(self, mol) -> bool:
        """Identify common peptide coupling reagents and activators."""
        # Common coupling reagents patterns
        coupling_reagents = [
            "[C](=[N])=[N]",  # Carbodiimides (EDC, DCC)
            "[P](=[O])",      # Phosphorus-based (PyBOP, HBTU)
            "[N]=[N]",        # Azide-based
            "[Cl][C](=[O])",  # Acid chlorides (simple ones)
        ]
        
        for pattern in coupling_reagents:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
                
        return False
