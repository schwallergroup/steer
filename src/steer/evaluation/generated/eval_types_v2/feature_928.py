"""Generated evaluation code for: Diels-Alder bicyclic core formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DielsAlderBicyclicCore(BaseScoring):
    """
    Evaluates synthesis routes for Diels-Alder reactions that form bicyclic products.
    Favors early-stage formation of bicyclic cyclohexene cores via Diels-Alder cycloaddition.
    """
    
    def __init__(self, config: Dict):
        self.stage = config["parameters"]["stage"]  # "early" preferred
        self.products_bicyclic = config["parameters"]["products_bicyclic"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Diels-Alder not found
        
        if self.stage == "early":
            return 1 - x  # Earlier is better (closer to 1.0)
        else:
            return x  # Later is better (closer to 1.0)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Diels-Alder forming bicyclic products"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
            
        reactants_smiles, products_smiles = rxn_smiles.split(">>")
        
        # Parse reactants and products
        reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
        
        if not all(reactants) or not all(products):
            return False
            
        # Check if this is a Diels-Alder reaction pattern
        if not self._is_diels_alder_reaction(reactants):
            return False
            
        # Check if products contain bicyclic structures
        if self.products_bicyclic:
            return any(self._has_bicyclic_structure(prod) for prod in products)
        
        return True
    
    def _is_diels_alder_reaction(self, reactants) -> bool:
        """Detect Diels-Alder reaction pattern: diene + dienophile -> cyclohexene"""
        if len(reactants) < 2:
            return False
            
        # SMARTS patterns for diene and dienophile
        diene_pattern = Chem.MolFromSmarts("[C,c]=[C,c]-[C,c]=[C,c]")  # Conjugated diene
        dienophile_pattern = Chem.MolFromSmarts("[C,c]=[C,c]")  # Alkene dienophile
        
        has_diene = False
        has_dienophile = False
        
        for mol in reactants:
            if mol.HasSubstructMatch(diene_pattern):
                has_diene = True
            elif mol.HasSubstructMatch(dienophile_pattern):
                has_dienophile = True
                
        return has_diene and has_dienophile
    
    def _has_bicyclic_structure(self, mol) -> bool:
        """Check if molecule contains bicyclic ring system"""
        if not mol:
            return False
            
        # Get ring information
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()
        
        if len(rings) < 2:
            return False
            
        # Check for fused or bridged bicyclic systems
        for i, ring1 in enumerate(rings):
            for ring2 in rings[i+1:]:
                shared_atoms = set(ring1) & set(ring2)
                if len(shared_atoms) >= 2:  # Fused rings share 2+ atoms
                    return True
                elif len(shared_atoms) == 1:  # Spiro center
                    return True
                    
        # Check for bridged bicyclic (atoms belonging to 3+ rings)
        atom_ring_count = [0] * mol.GetNumAtoms()
        for ring in rings:
            for atom_idx in ring:
                atom_ring_count[atom_idx] += 1
                
        if any(count >= 3 for count in atom_ring_count):
            return True
            
        return False
