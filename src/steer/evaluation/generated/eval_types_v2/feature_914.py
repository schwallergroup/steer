"""Generated evaluation code for: Convergent synthesis via Suzuki coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiCoupling(BaseScoring):
    """
    Evaluates convergent synthesis routes that use Suzuki-Miyaura coupling.
    Checks for the presence of Suzuki coupling reactions at mid-route stages
    where two substantial fragments are joined to form biaryl linkages.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.stage = config.get("stage", "mid_route")
        
        # Suzuki coupling typically involves boronic acid/ester + aryl halide
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC([#6])([#6])C([#6])([#6])O1")
        self.aryl_halide_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I]")
        self.biaryl_pattern = Chem.MolFromSmarts("c-c")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling not found
        
        if self.stage == "mid_route":
            # Prefer coupling at intermediate depths (0.3-0.7)
            if 0.3 <= x <= 0.7:
                return 1.0
            elif x < 0.3:
                return 0.5  # Too early
            else:
                return 0.3  # Too late
        elif self.stage == "early":
            return 1.0 - x  # Earlier is better
        else:  # late stage
            return x  # Later is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a Suzuki coupling forming a convergent junction.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        rxn_parts = mapped_rxn.split(">>")
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or len(reactant_mols) < 2:
                return False
            
            # Check if this is a convergent reaction (multiple substantial fragments)
            if not self._is_convergent_reaction(reactant_mols):
                return False
            
            # Check for Suzuki coupling pattern
            return self._is_suzuki_coupling(reactant_mols, product_mol)
            
        except:
            return False
    
    def _is_convergent_reaction(self, reactant_mols) -> bool:
        """
        Check if reaction involves multiple substantial fragments (convergent).
        """
        substantial_fragments = 0
        min_heavy_atoms = 6  # Minimum size for a substantial fragment
        
        for mol in reactant_mols:
            if mol and mol.GetNumHeavyAtoms() >= min_heavy_atoms:
                substantial_fragments += 1
        
        return substantial_fragments >= self.fragment_count
    
    def _is_suzuki_coupling(self, reactant_mols, product_mol) -> bool:
        """
        Check for characteristic Suzuki coupling patterns:
        - One reactant has boronic acid/ester
        - Another has aryl halide
        - Product forms new C-C bond (biaryl)
        """
        has_boronic_component = False
        has_aryl_halide = False
        
        for mol in reactant_mols:
            if not mol:
                continue
                
            # Check for boronic acid or ester
            if (mol.HasSubstructMatch(self.boronic_acid_pattern) or 
                mol.HasSubstructMatch(self.boronic_ester_pattern)):
                has_boronic_component = True
            
            # Check for aryl halide
            if mol.HasSubstructMatch(self.aryl_halide_pattern):
                has_aryl_halide = True
        
        # Check if product has more C-C bonds than individual reactants
        # (indicating new biaryl formation)
        if has_boronic_component and has_aryl_halide:
            return self._has_new_biaryl_bond(reactant_mols, product_mol)
        
        return False
    
    def _has_new_biaryl_bond(self, reactant_mols, product_mol) -> bool:
        """
        Check if the product has formed new aromatic C-C bonds.
        """
        if not product_mol:
            return False
        
        # Count aromatic-aromatic C-C bonds in product
        product_biaryl_bonds = 0
        for bond in product_mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if (atom1.GetIsAromatic() and atom2.GetIsAromatic() and 
                atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6):
                product_biaryl_bonds += 1
        
        # Count total aromatic-aromatic C-C bonds in reactants
        reactant_biaryl_bonds = 0
        for mol in reactant_mols:
            if mol:
                for bond in mol.GetBonds():
                    atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                    if (atom1.GetIsAromatic() and atom2.GetIsAromatic() and 
                        atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6):
                        reactant_biaryl_bonds += 1
        
        # New biaryl bond formed if product has more than reactants
        return product_biaryl_bonds > reactant_biaryl_bonds
