"""Generated evaluation code for: C-S bond formation via mesylate displacement"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MesylateDisplacementCSBond(BaseScoring):
    """
    Evaluates synthesis routes for C-S bond formation via mesylate displacement.
    
    Detects SN2 reactions where a thiolate nucleophile displaces a mesylate 
    leaving group to form a C-S bond. Returns better scores for earlier 
    occurrence of this key transformation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            return 1 - x  # Earlier occurrence is better
    
    def hit_condition(self, d) -> bool:
        """Check if reaction involves C-S bond formation via mesylate displacement."""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for mesylate leaving group in reactants
            mesylate_pattern = Chem.MolFromSmarts("[CH3:1][S:2](=[O:3])(=[O:4])[O:5][C:6]")
            has_mesylate = any(mol.HasSubstructMatch(mesylate_pattern) for mol in reactants)
            
            if not has_mesylate:
                return False
            
            # Check for thiolate/thiol nucleophile in reactants
            thiol_patterns = [
                Chem.MolFromSmarts("[S-:1]"),  # Thiolate anion
                Chem.MolFromSmarts("[SH1:1]"),  # Free thiol
                Chem.MolFromSmarts("[S:1][C,c]")  # Thioether as nucleophile precursor
            ]
            
            has_sulfur_nucleophile = any(
                any(mol.HasSubstructMatch(pattern) for mol in reactants)
                for pattern in thiol_patterns
            )
            
            if not has_sulfur_nucleophile:
                return False
            
            # Check for C-S bond formation (new C-S bond in products)
            cs_bond_pattern = Chem.MolFromSmarts("[C:1][S:2]")
            
            # Count C-S bonds in reactants vs products
            reactant_cs_bonds = sum(
                len(mol.GetSubstructMatches(cs_bond_pattern)) 
                for mol in reactants
            )
            product_cs_bonds = sum(
                len(mol.GetSubstructMatches(cs_bond_pattern)) 
                for mol in products
            )
            
            # New C-S bond should be formed
            cs_bond_formed = product_cs_bonds > reactant_cs_bonds
            
            # Check for mesylate departure (mesyl group should be in products as leaving group)
            mesyl_departure_pattern = Chem.MolFromSmarts("[CH3:1][S:2](=[O:3])(=[O:4])[OH1:5]")
            has_mesyl_product = any(mol.HasSubstructMatch(mesyl_departure_pattern) for mol in products)
            
            return cs_bond_formed and has_mesyl_product
            
        except Exception:
            return False
