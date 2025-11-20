"""Generated evaluation code for: Late stage Suzuki coupling assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki-Miyaura coupling reaction occurs at the final stage
    of synthesis, specifically targeting aryl-heteroaryl bond formation.
    """
    
    def __init__(self, config: Dict):
        self.stage = config.get("stage", "final")
        self.bond_type = config.get("bond_formed", "aryl-heteroaryl")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        elif self.stage == "final" and x > 0.9:
            return 10  # Perfect score for final stage
        elif self.stage == "final":
            return x * 10  # Scale depth fraction to 0-10
        else:
            return 1 - x  # Earlier stages get lower scores
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is a Suzuki coupling forming aryl-heteroaryl bond"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        # Check for Suzuki coupling pattern
        if not self._is_suzuki_coupling(mapped_rxn):
            return False
            
        # Check for aryl-heteroaryl bond formation
        if self.bond_type == "aryl-heteroaryl":
            return self._forms_aryl_heteroaryl_bond(mapped_rxn)
        
        return True
    
    def _is_suzuki_coupling(self, mapped_rxn: str) -> bool:
        """Detect Suzuki coupling by looking for boronic acid/ester reactants and Pd catalyst patterns"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[1].split(".")
            
            # Look for boronic acid or boronic ester pattern
            boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(-O)(-O)")  # R-B(OH)2
            boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1-O-C-C-O-1")  # Boronic ester
            pinacol_ester_pattern = Chem.MolFromSmarts("[#6]-B(-O-C(-C)(-C)-C(-C)(-C)-O)")  # Pinacol ester
            
            # Look for halide pattern (typically Br or I)
            aryl_halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")
            
            has_boron_reagent = False
            has_halide = False
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol is None:
                    continue
                    
                # Check for boron reagents
                if (mol.HasSubstructMatch(boronic_acid_pattern) or 
                    mol.HasSubstructMatch(boronic_ester_pattern) or
                    mol.HasSubstructMatch(pinacol_ester_pattern)):
                    has_boron_reagent = True
                    
                # Check for aryl halide
                if mol.HasSubstructMatch(aryl_halide_pattern):
                    has_halide = True
            
            return has_boron_reagent and has_halide
            
        except Exception:
            return False
    
    def _forms_aryl_heteroaryl_bond(self, mapped_rxn: str) -> bool:
        """Check if the reaction forms a bond between aromatic carbon and heteroaromatic carbon"""
        try:
            rxn_parts = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if product is None:
                return False
            
            # Get all aromatic carbons in product
            aromatic_carbons = []
            heteroaromatic_carbons = []
            
            for atom in product.GetAtoms():
                if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
                    # Check if this carbon is in a heteroaromatic ring
                    in_heteroaromatic = False
                    for ring in product.GetRingInfo().AtomRings():
                        if atom.GetIdx() in ring:
                            ring_atoms = [product.GetAtomWithIdx(i) for i in ring]
                            if any(a.GetSymbol() != 'C' for a in ring_atoms):
                                in_heteroaromatic = True
                                break
                    
                    if in_heteroaromatic:
                        heteroaromatic_carbons.append(atom.GetAtomMapNum())
                    else:
                        aromatic_carbons.append(atom.GetAtomMapNum())
            
            # Check if there's a bond between aromatic and heteroaromatic carbons
            # that wasn't present in reactants
            for bond in product.GetBonds():
                atom1_map = bond.GetBeginAtom().GetAtomMapNum()
                atom2_map = bond.GetEndAtom().GetAtomMapNum()
                
                # Check if this is an aryl-heteroaryl bond
                if ((atom1_map in aromatic_carbons and atom2_map in heteroaromatic_carbons) or
                    (atom1_map in heteroaromatic_carbons and atom2_map in aromatic_carbons)):
                    
                    # Verify this bond wasn't in reactants
                    bond_exists_in_reactants = False
                    for reactant in reactants:
                        if reactant is None:
                            continue
                        reactant_maps = [a.GetAtomMapNum() for a in reactant.GetAtoms()]
                        if atom1_map in reactant_maps and atom2_map in reactant_maps:
                            bond_exists_in_reactants = True
                            break
                    
                    if not bond_exists_in_reactants:
                        return True
            
            return False
            
        except Exception:
            return False
