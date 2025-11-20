"""Generated evaluation code for: Sonogashira coupling for alkyne installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SonogashiraCoupling(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sonogashira coupling reactions
    that install alkyne groups. Rewards routes where this reaction occurs earlier.
    """
    
    def __init__(self, config: Dict):
        self.reaction_type = config["parameters"]["reaction_type"]
        self.purpose = config["parameters"]["purpose"]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sonogashira coupling doesn't occur
        else:
            return 1 - x  # Earlier occurrence is better
    
    def hit_condition(self, d) -> bool:
        """
        Detects Sonogashira coupling by looking for:
        1. Formation of C-C triple bonds (alkynes)
        2. Typical reaction pattern: aryl/vinyl halide + terminal alkyne -> internal alkyne
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if an alkyne bond is formed
            if not self._has_alkyne_formation(product_mol, reactant_mols):
                return False
                
            # Check for typical Sonogashira pattern: halide + terminal alkyne
            return self._has_sonogashira_pattern(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _has_alkyne_formation(self, product, reactants):
        """Check if a new C≡C bond is formed in the product"""
        # Count alkynes in product
        alkyne_pattern = Chem.MolFromSmarts("C#C")
        product_alkynes = len(product.GetSubstructMatches(alkyne_pattern))
        
        # Count alkynes in reactants
        reactant_alkynes = sum(len(mol.GetSubstructMatches(alkyne_pattern)) for mol in reactants)
        
        # New alkyne bond formed or alkyne connectivity changed
        return product_alkynes > 0 and (product_alkynes >= reactant_alkynes)
    
    def _has_sonogashira_pattern(self, product, reactants):
        """Check for typical Sonogashira coupling patterns"""
        # Look for aryl/vinyl halides in reactants
        halide_patterns = [
            Chem.MolFromSmarts("c-[Cl,Br,I]"),  # aryl halide
            Chem.MolFromSmarts("C=C-[Cl,Br,I]"),  # vinyl halide
            Chem.MolFromSmarts("[Cl,Br,I]")  # any halide
        ]
        
        # Look for terminal alkyne in reactants
        terminal_alkyne_pattern = Chem.MolFromSmarts("C#C[H]")
        
        has_halide = any(
            any(mol.HasSubstructMatch(pattern) for pattern in halide_patterns)
            for mol in reactants
        )
        
        has_terminal_alkyne = any(
            mol.HasSubstructMatch(terminal_alkyne_pattern)
            for mol in reactants
        )
        
        # Look for internal alkyne in product (typical Sonogashira product)
        internal_alkyne_patterns = [
            Chem.MolFromSmarts("c-C#C"),  # aryl-alkyne
            Chem.MolFromSmarts("C-C#C-C"),  # internal alkyne
        ]
        
        has_internal_alkyne = any(
            product.HasSubstructMatch(pattern)
            for pattern in internal_alkyne_patterns
        )
        
        return has_halide and (has_terminal_alkyne or has_internal_alkyne)
