"""Generated evaluation code for: Early stage Diels-Alder cycloaddition for bicyclic core"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DielsAlderCoreFormation(BaseScoring):
    """
    Evaluates synthesis routes for early-stage Diels-Alder cycloaddition reactions
    that form bicyclic core scaffolds. Rewards routes where Diels-Alder reactions
    occur early in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing = config["parameters"]["timing"]  # "early"
        self.forms_core = config["parameters"]["forms_core_scaffold"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Diels-Alder reaction found
        
        if self.timing == "early":
            # Reward early occurrence - closer to 0 is better
            return max(0, 10 * (1 - x))
        else:
            # For other timing preferences, could implement different scoring
            return 5.0
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Diels-Alder cycloaddition forming bicyclic core"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            if not reactants or not products:
                return False
            
            # Check if this is a Diels-Alder reaction
            if not self._is_diels_alder_reaction(reactants, products):
                return False
            
            # Check if it forms a bicyclic core scaffold
            if self.forms_core and not self._forms_bicyclic_core(products):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _is_diels_alder_reaction(self, reactants, products) -> bool:
        """Detect Diels-Alder reaction pattern"""
        # Look for formation of 6-membered rings from reactants
        reactant_ring_count = sum(len([ring for ring in r.GetRingInfo().AtomRings() 
                                     if len(ring) == 6]) for r in reactants)
        product_ring_count = sum(len([ring for ring in p.GetRingInfo().AtomRings() 
                                    if len(ring) == 6]) for p in products)
        
        # Diels-Alder should form at least one new 6-membered ring
        if product_ring_count <= reactant_ring_count:
            return False
        
        # Check for diene pattern (conjugated C=C-C=C)
        diene_pattern = Chem.MolFromSmarts("[C]=[C]-[C]=[C]")
        has_diene = any(r.HasSubstructMatch(diene_pattern) for r in reactants)
        
        # Check for dienophile pattern (C=C, C=O, C=N, etc.)
        dienophile_patterns = [
            Chem.MolFromSmarts("[C]=[C]"),  # alkene
            Chem.MolFromSmarts("[C]=[O]"),  # carbonyl
            Chem.MolFromSmarts("[C]=[N]"),  # imine
            Chem.MolFromSmarts("[C]#[C]"),  # alkyne
            Chem.MolFromSmarts("[C]#[N]")   # nitrile
        ]
        
        has_dienophile = any(
            any(r.HasSubstructMatch(pattern) for r in reactants)
            for pattern in dienophile_patterns
        )
        
        return has_diene and has_dienophile
    
    def _forms_bicyclic_core(self, products) -> bool:
        """Check if products contain bicyclic structures"""
        for product in products:
            ring_info = product.GetRingInfo()
            if ring_info.NumRings() >= 2:
                # Check for fused or bridged bicyclic systems
                atom_rings = ring_info.AtomRings()
                for i, ring1 in enumerate(atom_rings):
                    for ring2 in atom_rings[i+1:]:
                        # If rings share atoms, they form a bicyclic system
                        if len(set(ring1) & set(ring2)) > 0:
                            return True
        return False
