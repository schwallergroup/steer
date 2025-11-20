"""Generated evaluation code for: Late stage trichloroacetimidate ether coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TrichloroacetimidateCoupling(BaseScoring):
    """
    Evaluates synthesis routes for late-stage trichloroacetimidate ether coupling reactions.
    Rewards routes where C-O bond formation occurs via trichloroacetimidate activation
    at later stages of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Late-stage coupling is better (lower depth fraction preferred)
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Penalize early occurrence, reward late occurrence
                return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Detects trichloroacetimidate ether coupling by checking for:
        1. Trichloroacetimidate leaving group (CCl3-C(=N)-O-)
        2. Formation of new C-O bond
        3. Loss of the acetimidate group
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for trichloroacetimidate pattern in reactants
            tca_pattern = Chem.MolFromSmarts("[C](Cl)(Cl)(Cl)-[C](=[N])-[O]-[C,c]")
            has_tca_reactant = any(mol.HasSubstructMatch(tca_pattern) for mol in reactants if mol)
            
            if not has_tca_reactant:
                return False
            
            # Check for acetimidate byproduct formation
            # Pattern for trichloroacetamide or related leaving group
            leaving_group_pattern = Chem.MolFromSmarts("[C](Cl)(Cl)(Cl)-[C](=[N])")
            has_leaving_group = any(mol.HasSubstructMatch(leaving_group_pattern) for mol in products if mol)
            
            # Alternative: check for HN=C(CCl3) pattern
            alt_leaving_pattern = Chem.MolFromSmarts("[N]=[C]-[C](Cl)(Cl)(Cl)")
            has_alt_leaving = any(mol.HasSubstructMatch(alt_leaving_pattern) for mol in products if mol)
            
            # Count C-O bonds in reactants vs products to confirm new ether formation
            reactant_co_bonds = sum(self._count_co_bonds(mol) for mol in reactants if mol)
            product_co_bonds = sum(self._count_co_bonds(mol) for mol in products if mol)
            
            # Should have net formation of C-O bonds (accounting for the one broken from TCA)
            co_bond_change = product_co_bonds - reactant_co_bonds
            
            return has_tca_reactant and (has_leaving_group or has_alt_leaving) and co_bond_change >= -1
            
        except Exception:
            return False
    
    def _count_co_bonds(self, mol) -> int:
        """Count C-O single bonds in molecule (excluding carbonyl)"""
        if not mol:
            return 0
        count = 0
        for bond in mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if bond.GetBondType() == Chem.BondType.SINGLE:
                symbols = sorted([atom1.GetSymbol(), atom2.GetSymbol()])
                if symbols == ['C', 'O']:
                    count += 1
        return count
