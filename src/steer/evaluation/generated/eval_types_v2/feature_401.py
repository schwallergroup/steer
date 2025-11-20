"""Generated evaluation code for: Early stage biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyStageBiarylFormation(BaseScoring):
    """
    Evaluates whether biaryl formation (specifically Suzuki-Miyaura coupling) 
    occurs early in the synthesis route. Returns higher scores for earlier 
    biaryl bond formation.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config["parameters"]["bond_type"]
        self.timing = config["parameters"]["timing"] 
        self.reaction_type = config["parameters"]["reaction_type"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No biaryl formation detected
        else:
            # Early stage formation gets higher score (inverse of depth fraction)
            return 1 - x
            
    def hit_condition(self, d) -> bool:
        """Check if this reaction forms a biaryl bond via Suzuki coupling"""
        metadata = d.get("metadata", {})
        
        # First check if this looks like a Suzuki reaction
        if not self._is_suzuki_reaction(d):
            return False
            
        # Then check if a biaryl bond is formed
        return self._forms_biaryl_bond(d)
        
    def _is_suzuki_reaction(self, d) -> bool:
        """Detect Suzuki-Miyaura coupling pattern"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        reactants = parts[1].split(".")
        
        # Look for boronic acid/ester pattern and halide pattern
        has_boron = False
        has_halide = False
        
        for reactant in reactants:
            mol = Chem.MolFromSmiles(reactant)
            if mol is None:
                continue
                
            # Check for boronic acid/ester: B(O)(O) or B(OC)(OC)
            boron_pattern1 = Chem.MolFromSmarts("[#5](O)(O)")  # B(OH)2
            boron_pattern2 = Chem.MolFromSmarts("[#5]1OC[CH2]O1")  # cyclic boronate
            boron_pattern3 = Chem.MolFromSmarts("[#5](OC)(OC)")  # B(OR)2
            
            if (mol.HasSubstructMatch(boron_pattern1) or 
                mol.HasSubstructMatch(boron_pattern2) or
                mol.HasSubstructMatch(boron_pattern3)):
                has_boron = True
                
            # Check for aryl halide
            halide_pattern = Chem.MolFromSmarts("c[Cl,Br,I]")
            if mol.HasSubstructMatch(halide_pattern):
                has_halide = True
                
        return has_boron and has_halide
        
    def _forms_biaryl_bond(self, d) -> bool:
        """Check if the reaction forms a new biaryl bond"""
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles:
            return False
            
        parts = rxn_smiles.split(">>")
        if len(parts) != 2:
            return False
            
        product = Chem.MolFromSmiles(parts[0])
        reactants = [Chem.MolFromSmiles(r) for r in parts[1].split(".")]
        
        if product is None or any(r is None for r in reactants):
            return False
            
        # Count biaryl bonds (aromatic C-C bonds between rings) in product vs reactants
        product_biaryl_count = self._count_biaryl_bonds(product)
        reactant_biaryl_count = sum(self._count_biaryl_bonds(r) for r in reactants)
        
        # New biaryl bond formed if product has more than reactants
        return product_biaryl_count > reactant_biaryl_count
        
    def _count_biaryl_bonds(self, mol) -> int:
        """Count the number of aromatic C-C bonds between different aromatic rings"""
        if mol is None:
            return 0
            
        count = 0
        ri = mol.GetRingInfo()
        
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Both atoms must be aromatic carbons
            if (atom1.GetAtomicNum() == 6 and atom1.GetIsAromatic() and
                atom2.GetAtomicNum() == 6 and atom2.GetIsAromatic()):
                
                # Check if atoms are in different aromatic rings
                atom1_rings = [r for r in ri.AtomRings() if atom1.GetIdx() in r and len(r) >= 5]
                atom2_rings = [r for r in ri.AtomRings() if atom2.GetIdx() in r and len(r) >= 5]
                
                # If atoms are in different rings, this is a biaryl bond
                if atom1_rings and atom2_rings:
                    in_same_ring = any(ring1 == ring2 for ring1 in atom1_rings for ring2 in atom2_rings)
                    if not in_same_ring:
                        count += 1
                        
        return count
