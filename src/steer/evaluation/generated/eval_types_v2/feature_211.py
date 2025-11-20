"""Generated evaluation code for: Cbz protecting group for amine selectivity"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CbzAmineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of Cbz (benzyloxycarbonyl) protecting groups 
    on amines for chemoselectivity purposes. Returns higher scores when Cbz protection 
    occurs earlier in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No Cbz protection found
        else:
            return 1 - x  # Earlier protection is better
    
    def hit_condition(self, d):
        """Check if this reaction introduces a Cbz protecting group on an amine"""
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if Cbz group is formed (look for Cbz reagent in reactants)
            cbz_reagent_pattern = Chem.MolFromSmarts("O=C(OCC1=CC=CC=C1)O[CH]")  # Cbz-Cl or similar
            cbz_reagent_alt = Chem.MolFromSmarts("ClC(=O)OCc1ccccc1")  # Cbz-Cl specifically
            
            has_cbz_reagent = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(cbz_reagent_pattern) or reactant.HasSubstructMatch(cbz_reagent_alt):
                    has_cbz_reagent = True
                    break
            
            if not has_cbz_reagent:
                return False
            
            # Check if an amine gets protected (Cbz-NH pattern appears in products)
            cbz_protected_amine = Chem.MolFromSmarts("O=C(OCc1ccccc1)N")  # Cbz-NH pattern
            
            has_protected_amine_product = False
            for product in products:
                if product.HasSubstructMatch(cbz_protected_amine):
                    has_protected_amine_product = True
                    break
            
            # Check that reactants contain free amine
            free_amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")  # Primary or secondary amine
            has_free_amine_reactant = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(free_amine_pattern):
                    has_free_amine_reactant = True
                    break
            
            return has_cbz_reagent and has_protected_amine_product and has_free_amine_reactant
            
        except Exception:
            return False
