"""Generated evaluation code for: Early ring-closing metathesis for bicyclic core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyRingClosingMetathesis(BaseScoring):
    """
    Evaluates whether ring-closing metathesis occurs early in the synthesis route.
    
    Ring-closing metathesis (RCM) is identified by:
    - Formation of C=C double bonds in cyclic structures
    - Presence of alkene substrates that form rings
    - Detection of metathesis-like transformations
    
    Early timing is defined as occurring before the stage_threshold fraction
    of the total synthesis depth.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"]["stage_threshold"]
        
    def route_scoring(self, x) -> float:
        """
        Score based on how early the RCM occurs.
        
        Args:
            x: Depth fraction where RCM occurs (-1 if not found)
            
        Returns:
            Score from 0-1 (higher is better for earlier RCM)
        """
        if x < 0:
            return 0  # RCM not found
        
        if x <= self.stage_threshold:
            # Early RCM gets high score, linearly decreasing as it gets later
            return 1.0 - (x / self.stage_threshold) * 0.3
        else:
            # Late RCM gets lower score
            return 0.3 * (1.0 - x) / (1.0 - self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents ring-closing metathesis.
        
        Args:
            d: Reaction node dictionary with metadata
            
        Returns:
            True if this reaction is likely RCM
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for RCM pattern: 
            # 1. Reactant should have terminal alkenes that can cyclize
            # 2. Product should have new ring with C=C bond
            # 3. Should lose ethylene or similar small alkene
            
            return (self._has_rcm_substrate_pattern(reactants) and 
                    self._forms_cyclic_alkene(reactants, products) and
                    self._has_metathesis_byproduct(reactants, products))
                    
        except Exception:
            return False
    
    def _has_rcm_substrate_pattern(self, reactants) -> bool:
        """Check if reactants contain RCM substrate pattern (diene that can cyclize)"""
        for mol in reactants:
            # Look for molecules with two terminal alkenes
            terminal_alkene = Chem.MolFromSmarts("C=C")
            if mol.HasSubstructMatch(terminal_alkene):
                matches = mol.GetSubstructMatches(terminal_alkene)
                if len(matches) >= 2:
                    # Check if the alkenes could feasibly form a ring
                    return True
        return False
    
    def _forms_cyclic_alkene(self, reactants, products) -> bool:
        """Check if products have more rings and internal C=C than reactants"""
        reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants)
        product_rings = sum(mol.GetRingInfo().NumRings() for mol in products)
        
        # Should form at least one new ring
        if product_rings <= reactant_rings:
            return False
            
        # Check for internal alkene in ring
        for mol in products:
            ring_info = mol.GetRingInfo()
            for ring in ring_info.AtomRings():
                for bond in mol.GetBonds():
                    if (bond.GetBondType() == Chem.BondType.DOUBLE and
                        bond.GetBeginAtomIdx() in ring and 
                        bond.GetEndAtomIdx() in ring):
                        return True
        return False
    
    def _has_metathesis_byproduct(self, reactants, products) -> bool:
        """Check for typical metathesis byproducts like ethylene"""
        # Count carbons in reactants vs products
        reactant_carbons = sum(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C') 
                              for mol in reactants)
        product_carbons = sum(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C') 
                             for mol in products)
        
        # Should lose some carbons (typically 2 for ethylene loss)
        carbon_loss = reactant_carbons - product_carbons
        return 2 <= carbon_loss <= 6  # Reasonable range for metathesis byproducts
