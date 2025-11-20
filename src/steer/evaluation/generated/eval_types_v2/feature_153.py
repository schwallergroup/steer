"""Generated evaluation code for: Late stage Buchwald-Hartwig coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BuchwaldHartwigCoupling(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Buchwald-Hartwig coupling reactions
    at a specific timing in the synthesis. Buchwald-Hartwig reactions form C-N bonds
    between aryl halides and amines using palladium catalysis.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late-stage")
        self.step_position = config.get("step_position", 3)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "late-stage":
            return 1 - x  # Later is better (lower depth fraction)
        elif self.timing == "mid-stage":
            # Optimal around middle, penalize very early or very late
            optimal_fraction = 0.5
            return 1 - abs(x - optimal_fraction)
        else:  # early-stage
            return x  # Earlier is better (higher depth fraction)
    
    def hit_condition(self, d):
        """
        Detects Buchwald-Hartwig coupling by looking for:
        1. Formation of C-N bond between aromatic carbon and nitrogen
        2. Presence of aryl halide reactant and amine reactant
        3. Characteristic bond formation pattern
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
            
            # Check for C-N bond formation between aromatic systems
            return self._detect_buchwald_hartwig_pattern(product_mol, reactant_mols)
            
        except Exception:
            return False
    
    def _detect_buchwald_hartwig_pattern(self, product, reactants):
        """
        Detects the characteristic pattern of Buchwald-Hartwig coupling:
        - Aryl halide + amine -> aryl amine
        """
        # Look for aromatic C-N bonds in product
        aromatic_cn_bonds = []
        for bond in product.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if ((atom1.GetAtomicNum() == 6 and atom1.GetIsAromatic() and 
                 atom2.GetAtomicNum() == 7) or
                (atom2.GetAtomicNum() == 6 and atom2.GetIsAromatic() and 
                 atom1.GetAtomicNum() == 7)):
                aromatic_cn_bonds.append((atom1.GetAtomMapNum(), atom2.GetAtomMapNum()))
        
        if not aromatic_cn_bonds:
            return False
        
        # Check if reactants contain aryl halide and amine patterns
        has_aryl_halide = False
        has_amine = False
        
        for reactant in reactants:
            # Check for aryl halide (aromatic carbon bonded to halogen)
            aryl_halide_pattern = Chem.MolFromSmarts("[cH0,c:1]-[F,Cl,Br,I]")
            if reactant.HasSubstructMatch(aryl_halide_pattern):
                has_aryl_halide = True
            
            # Check for amine (nitrogen with at least one hydrogen or alkyl)
            amine_pattern = Chem.MolFromSmarts("[N;H1,H2:1]")
            secondary_amine_pattern = Chem.MolFromSmarts("[N;H1:1]([C,c])")
            if (reactant.HasSubstructMatch(amine_pattern) or 
                reactant.HasSubstructMatch(secondary_amine_pattern)):
                has_amine = True
        
        # Additional check: verify that the C-N bond is newly formed
        if has_aryl_halide and has_amine:
            return self._verify_bond_formation(aromatic_cn_bonds, reactants)
        
        return False
    
    def _verify_bond_formation(self, cn_bonds, reactants):
        """
        Verify that the C-N bond is newly formed by checking it doesn't exist in reactants
        """
        for c_map, n_map in cn_bonds:
            if c_map and n_map:  # Both atoms have map numbers
                # Check if this C-N bond exists in any reactant
                bond_exists_in_reactants = False
                for reactant in reactants:
                    c_atom = None
                    n_atom = None
                    for atom in reactant.GetAtoms():
                        if atom.GetAtomMapNum() == c_map:
                            c_atom = atom
                        elif atom.GetAtomMapNum() == n_map:
                            n_atom = atom
                    
                    if c_atom and n_atom:
                        # Both atoms in same reactant, check if bonded
                        if reactant.GetBondBetweenAtoms(c_atom.GetIdx(), n_atom.GetIdx()):
                            bond_exists_in_reactants = True
                            break
                
                if not bond_exists_in_reactants:
                    return True  # New C-N bond formed
        
        return False
