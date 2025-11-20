"""Generated evaluation code for: Late stage benzylic methyl oxidation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageBenzylicMethylOxidation(BaseScoring):
    """
    Evaluates routes for late-stage benzylic methyl oxidation reactions.
    Detects oxidation of benzylic methyl groups (c-CH3) to alcohols (c-CH2OH)
    and scores based on how late in the synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.substrate_pattern = Chem.MolFromSmarts("c-[CH3]")
        self.product_pattern = Chem.MolFromSmarts("c-[CH2OH]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Oxidation doesn't happen
        else:
            return 1 - x  # Later oxidation is better (closer to 1.0)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction performs benzylic methyl oxidation.
        Returns True if c-CH3 in reactant becomes c-CH2OH in product.
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            # Check if any reactant has benzylic methyl pattern
            has_benzylic_methyl = any(r.HasSubstructMatch(self.substrate_pattern) for r in reactants)
            
            # Check if any product has benzylic alcohol pattern  
            has_benzylic_alcohol = any(p.HasSubstructMatch(self.product_pattern) for p in products)
            
            if not (has_benzylic_methyl and has_benzylic_alcohol):
                return False
            
            # Verify this is actually an oxidation by checking atom mapping
            return self._verify_oxidation_mapping(reactants, products)
            
        except (KeyError, ValueError, AttributeError):
            return False
    
    def _verify_oxidation_mapping(self, reactants, products):
        """
        Verify that a benzylic methyl carbon is actually oxidized to alcohol
        using atom mapping numbers.
        """
        # Find mapped benzylic methyl carbons in reactants
        benzylic_methyl_carbons = set()
        for reactant in reactants:
            if reactant.HasSubstructMatch(self.substrate_pattern):
                matches = reactant.GetSubstructMatches(self.substrate_pattern)
                for match in matches:
                    # match[1] should be the methyl carbon (c-[CH3])
                    methyl_carbon_idx = match[1]
                    atom = reactant.GetAtomWithIdx(methyl_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        benzylic_methyl_carbons.add(atom.GetAtomMapNum())
        
        # Find mapped benzylic alcohol carbons in products
        benzylic_alcohol_carbons = set()
        for product in products:
            if product.HasSubstructMatch(self.product_pattern):
                matches = product.GetSubstructMatches(self.product_pattern)
                for match in matches:
                    # match[1] should be the alcohol carbon (c-[CH2OH])
                    alcohol_carbon_idx = match[1]
                    atom = product.GetAtomWithIdx(alcohol_carbon_idx)
                    if atom.GetAtomMapNum() > 0:
                        benzylic_alcohol_carbons.add(atom.GetAtomMapNum())
        
        # Check if any benzylic methyl carbon became a benzylic alcohol carbon
        return len(benzylic_methyl_carbons.intersection(benzylic_alcohol_carbons)) > 0
