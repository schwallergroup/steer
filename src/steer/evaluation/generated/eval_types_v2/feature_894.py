"""Generated evaluation code for: Late stage C-C coupling for final assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCCCoupling(BaseScoring):
    """
    Evaluates whether a C-C coupling reaction occurs at a specific late stage,
    typically involving alkyl or aryl halides as coupling partners.
    """
    
    def __init__(self, config: Dict):
        self.step_from_end = config["parameters"].get("step_from_end", 1)
        self.coupling_partners = config["parameters"].get("coupling_partners", ["alkyl_halide", "aryl_halide"])
        
        # SMARTS patterns for coupling partners
        self.patterns = {
            "alkyl_halide": "[CX4][F,Cl,Br,I]",
            "aryl_halide": "[cX3][F,Cl,Br,I]",
            "organometallic": "[C][B,Zn,Mg,Sn]",
            "boronic_acid": "[C]B([OH])[OH]",
            "boronic_ester": "[C]B1OCCCO1"
        }
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # C-C coupling doesn't happen
        
        # Calculate expected position from end
        expected_position = 1.0 - (self.step_from_end / 10.0)  # Convert to fraction from start
        
        # Score based on how close the coupling is to the expected late stage
        if x >= expected_position:
            return 10  # Perfect late stage timing
        else:
            # Penalize early stage coupling
            return max(0, 10 * (x / expected_position))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents a C-C coupling with appropriate partners"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            products, reactants = mapped_rxn.split(">>")
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not react_mols:
                return False
            
            # Check if C-C bond is formed
            if not self._has_cc_bond_formation(prod_mol, react_mols):
                return False
            
            # Check for appropriate coupling partners
            return self._has_coupling_partners(react_mols)
            
        except Exception:
            return False
    
    def _has_cc_bond_formation(self, product, reactants) -> bool:
        """Check if a new C-C bond is formed in the reaction"""
        # Get all C-C bonds in product
        prod_cc_bonds = self._get_cc_bonds(product)
        
        # Get all C-C bonds in reactants
        reactant_cc_bonds = set()
        for reactant in reactants:
            reactant_cc_bonds.update(self._get_cc_bonds(reactant))
        
        # Check if there are new C-C bonds (considering atom mapping)
        new_cc_bonds = prod_cc_bonds - reactant_cc_bonds
        return len(new_cc_bonds) > 0
    
    def _get_cc_bonds(self, mol) -> set:
        """Get set of C-C bonds using atom map numbers"""
        cc_bonds = set()
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            if (atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C' and 
                atom1.GetAtomMapNum() > 0 and atom2.GetAtomMapNum() > 0):
                map1, map2 = atom1.GetAtomMapNum(), atom2.GetAtomMapNum()
                cc_bonds.add((min(map1, map2), max(map1, map2)))
        
        return cc_bonds
    
    def _has_coupling_partners(self, reactants) -> bool:
        """Check if reactants contain appropriate coupling partners"""
        partner_count = 0
        
        for reactant in reactants:
            if self._is_coupling_partner(reactant):
                partner_count += 1
                
        # Need at least 2 coupling partners or 1 if the other is an organometallic
        return partner_count >= 1
    
    def _is_coupling_partner(self, mol) -> bool:
        """Check if molecule matches coupling partner patterns"""
        for partner_type in self.coupling_partners:
            if partner_type in self.patterns:
                pattern = Chem.MolFromSmarts(self.patterns[partner_type])
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
        
        # Also check for common organometallic coupling partners
        for pattern_name in ["organometallic", "boronic_acid", "boronic_ester"]:
            pattern = Chem.MolFromSmarts(self.patterns[pattern_name])
            if pattern and mol.HasSubstructMatch(pattern):
                return True
                
        return False
