"""Generated evaluation code for: Late stage Suzuki coupling for backbone assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction occurs at a late stage (depth <= 4).
    Returns higher scores when Suzuki coupling happens closer to the target molecule.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 4)
        # Suzuki coupling SMARTS patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)C(C)(C)O1")  # Pinacol ester
        self.aryl_halide_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I]")
        self.vinyl_halide_pattern = Chem.MolFromSmarts("C=C[F,Cl,Br,I]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Late stage is better - normalize depth fraction to 0-1, then scale to 0-10
            if x <= self.depth_threshold / 10.0:  # x is depth fraction, so convert threshold
                return (1 - x) * 10  # Earlier = higher score
            else:
                return 0  # Too late in synthesis
    
    def hit_condition(self, d):
        """
        Check if this reaction is a Suzuki coupling by looking for:
        1. Boronic acid/ester + aryl/vinyl halide reactants
        2. C-C bond formation between aromatic/vinyl carbons
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            product_smiles, reactants_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for Suzuki reactant patterns
            has_boron_species = False
            has_halide_species = False
            
            for reactant in reactants:
                # Check for boronic acid or ester
                if (reactant.HasSubstructMatch(self.boronic_acid_pattern) or 
                    reactant.HasSubstructMatch(self.boronic_ester_pattern)):
                    has_boron_species = True
                
                # Check for aryl or vinyl halide
                if (reactant.HasSubstructMatch(self.aryl_halide_pattern) or 
                    reactant.HasSubstructMatch(self.vinyl_halide_pattern)):
                    has_halide_species = True
            
            # Must have both coupling partners
            if not (has_boron_species and has_halide_species):
                return False
            
            # Additional check: look for C-C bond formation
            # Count aromatic C-C bonds in product vs sum in reactants
            product_cc_bonds = self._count_aromatic_cc_bonds(product)
            reactant_cc_bonds = sum(self._count_aromatic_cc_bonds(r) for r in reactants)
            
            # Suzuki should form new C-C bond
            return product_cc_bonds > reactant_cc_bonds
            
        except Exception:
            return False
    
    def _count_aromatic_cc_bonds(self, mol):
        """Count C-C bonds involving at least one aromatic carbon"""
        if not mol:
            return 0
            
        count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Both atoms are carbon and at least one is aromatic
            if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 6 and 
                (atom1.GetIsAromatic() or atom2.GetIsAromatic())):
                count += 1
                
        return count
