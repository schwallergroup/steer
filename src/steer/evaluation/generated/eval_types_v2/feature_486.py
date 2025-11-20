"""Generated evaluation code for: Strategic halodesilylation for regiocontrol"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HalodesilylationRegiocontrol(BaseScoring):
    """
    Evaluates synthesis routes for strategic halodesilylation reactions where
    C-Si bonds are broken for regiocontrol purposes. TMS groups serve as
    placeholders to ensure regiospecific halogen introduction.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Earlier halodesilylation (lower depth) is generally better for regiocontrol.
        """
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier is better for strategic regiocontrol
            if self.condition_type == "bool":
                return 10 if x <= self.target_depth else 0
            else:
                # Score based on how close to target depth
                return max(0, 10 - 10 * abs(x - self.target_depth))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a halodesilylation reaction
        by detecting C-Si bond breaking with halogen introduction.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for silicon in reactants but not in main product
            has_si_reactant = any(self._has_silicon(mol) for mol in reactant_mols)
            main_product = product_mols[0]  # Assume first product is main product
            has_si_product = self._has_silicon(main_product)
            
            # Check for halogen introduction in product
            has_halogen_product = self._has_halogen(main_product)
            
            # Look for trimethylsilyl (TMS) pattern specifically
            tms_pattern = Chem.MolFromSmarts("[Si](C)(C)C")
            has_tms_reactant = any(mol.HasSubstructMatch(tms_pattern) for mol in reactant_mols if mol)
            
            # Halodesilylation: Si in reactant, halogen in product, Si removed from main product
            if has_si_reactant and has_halogen_product and not has_si_product and has_tms_reactant:
                # Additional check: verify C-Si bond breaking by atom mapping if available
                return self._verify_c_si_bond_break(d)
            
            return False
            
        except Exception:
            return False
    
    def _has_silicon(self, mol) -> bool:
        """Check if molecule contains silicon atoms."""
        if mol is None:
            return False
        return any(atom.GetAtomicNum() == 14 for atom in mol.GetAtoms())
    
    def _has_halogen(self, mol) -> bool:
        """Check if molecule contains halogen atoms (F, Cl, Br, I)."""
        if mol is None:
            return False
        halogen_nums = {9, 17, 35, 53}  # F, Cl, Br, I
        return any(atom.GetAtomicNum() in halogen_nums for atom in mol.GetAtoms())
    
    def _verify_c_si_bond_break(self, d) -> bool:
        """
        Verify C-Si bond breaking using atom mapping information.
        Check if carbon-silicon connectivity changes between reactants and products.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if ">>" not in rxn_smiles:
                return True  # Default to True if no mapping available
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Get atom mappings for C-Si bonds in reactants
            c_si_bonds_reactants = set()
            for mol in reactant_mols:
                if mol:
                    for bond in mol.GetBonds():
                        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                        if ((atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 14) or
                            (atom1.GetAtomicNum() == 14 and atom2.GetAtomicNum() == 6)):
                            map1, map2 = atom1.GetAtomMapNum(), atom2.GetAtomMapNum()
                            if map1 > 0 and map2 > 0:
                                c_si_bonds_reactants.add((min(map1, map2), max(map1, map2)))
            
            # Get atom mappings for C-Si bonds in products
            c_si_bonds_products = set()
            for mol in product_mols:
                if mol:
                    for bond in mol.GetBonds():
                        atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
                        if ((atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 14) or
                            (atom1.GetAtomicNum() == 14 and atom2.GetAtomicNum() == 6)):
                            map1, map2 = atom1.GetAtomMapNum(), atom2.GetAtomMapNum()
                            if map1 > 0 and map2 > 0:
                                c_si_bonds_products.add((min(map1, map2), max(map1, map2)))
            
            # Check if any C-Si bonds were broken
            broken_c_si_bonds = c_si_bonds_reactants - c_si_bonds_products
            return len(broken_c_si_bonds) > 0
            
        except Exception:
            return True  # Default to True if mapping analysis fails
