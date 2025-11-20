"""Generated evaluation code for: Late stage carbon-sulfur bond formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageCarbonSulfurBondFormation(BaseScoring):
    """
    Evaluates routes for late-stage carbon-sulfur bond formation.
    Rewards routes where C-S bonds are formed in the final synthetic steps.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config.get("bond_type", "C-S")
        self.timing = config.get("timing", "late")
        self.direction = config.get("direction", "formation")
    
    def route_scoring(self, x: float) -> float:
        if x < 0:
            return 0  # C-S bond formation doesn't happen
        else:
            # Late-stage formation is better, so higher depth fraction = higher score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves C-S bond formation by comparing
        C-S bond counts between reactants and products.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            # Filter out None molecules
            products = [mol for mol in products if mol is not None]
            reactants = [mol for mol in reactants if mol is not None]
            
            if not products or not reactants:
                return False
            
            # Count C-S bonds in products and reactants
            product_cs_bonds = sum(self._count_cs_bonds(mol) for mol in products)
            reactant_cs_bonds = sum(self._count_cs_bonds(mol) for mol in reactants)
            
            # Check for C-S bond formation (more C-S bonds in products than reactants)
            if self.direction == "formation":
                return product_cs_bonds > reactant_cs_bonds
            elif self.direction == "breaking":
                return reactant_cs_bonds > product_cs_bonds
            
        except Exception:
            return False
        
        return False
    
    def _count_cs_bonds(self, mol) -> int:
        """Count the number of carbon-sulfur bonds in a molecule."""
        if mol is None:
            return 0
        
        cs_count = 0
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            
            # Check if bond is between carbon and sulfur
            atoms = {atom1.GetSymbol(), atom2.GetSymbol()}
            if atoms == {"C", "S"}:
                cs_count += 1
        
        return cs_count
