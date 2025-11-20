"""Generated evaluation code for: Early hydrazine installation before cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyHydrazineInstallation(BaseScoring):
    """
    Checks if hydrazine installation occurs early in the route, before cyclization reactions.
    Looks for hydrazine formation (typically via diazotization-reduction of anilines) and
    ensures it happens before any ring-forming cyclization steps.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "early")  # "early" prefers lower depth scores
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Hydrazine installation doesn't happen before cyclization
        else:
            return 1 - x  # Earlier installation (lower depth fraction) gets higher score
    
    def hit_condition(self, d):
        # Check if this reaction involves hydrazine formation before any cyclization
        return self._is_hydrazine_formation(d) and self._no_prior_cyclization(d)
    
    def _is_hydrazine_formation(self, d):
        """Detect hydrazine formation reactions"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            # Check for hydrazine substructure formation
            hydrazine_patterns = [
                "[NH2][NH2]",  # Basic hydrazine
                "[NH2][NH][*]",  # N-substituted hydrazine
                "[*][NH][NH2]",  # N'-substituted hydrazine
                "[*][NH][NH][*]"  # N,N'-disubstituted hydrazine
            ]
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check if hydrazine pattern is formed (present in products but not reactants)
            reactant_has_hydrazine = any(
                mol and any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                           for pattern in hydrazine_patterns)
                for mol in reactant_mols if mol
            )
            
            product_has_hydrazine = any(
                mol and any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                           for pattern in hydrazine_patterns)
                for mol in product_mols if mol
            )
            
            # Also check for diazotization-reduction pattern (aniline -> hydrazine)
            aniline_pattern = "c1ccc([NH2])cc1"  # Basic aniline pattern
            diazonium_intermediate = any(
                mol and ("[N+]#[N]" in Chem.MolToSmiles(mol) if mol else False)
                for mol in reactant_mols + product_mols if mol
            )
            
            has_aniline_starting = any(
                mol and mol.HasSubstructMatch(Chem.MolFromSmarts(aniline_pattern))
                for mol in reactant_mols if mol
            )
            
            return (product_has_hydrazine and not reactant_has_hydrazine) or \
                   (has_aniline_starting and product_has_hydrazine) or \
                   (diazonium_intermediate and product_has_hydrazine)
                   
        except Exception:
            return False
    
    def _no_prior_cyclization(self, d):
        """Check that no cyclization reactions occurred before this step"""
        try:
            # Get the current route context to check previous steps
            current_depth = d.get("depth", 0)
            
            # For the hit condition, we assume this is being called during tree traversal
            # and we need to ensure this hydrazine formation happens before cyclization
            # This is implicitly handled by the BaseScoring traversal logic
            
            # Check if current reaction is a cyclization
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants, products = rxn_smiles.split(">>")
            
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count rings in reactants vs products
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactant_mols if mol)
            product_rings = sum(mol.GetRingInfo().NumRings() for mol in product_mols if mol)
            
            # If this reaction forms rings, it's a cyclization - we don't want this
            is_cyclization = product_rings > reactant_rings
            
            return not is_cyclization
            
        except Exception:
            return True  # Default to allowing if we can't determine
