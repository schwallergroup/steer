"""Generated evaluation code for: Nitrile to amidoxime conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NitrileToAmidoxime(BaseScoring):
    """
    Checks for nitrile to amidoxime conversion (C≡N to C=N-NOH).
    This transformation is a key step in oxadiazole synthesis routes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # SMARTS patterns for nitrile and amidoxime
        self.nitrile_pattern = Chem.MolFromSmarts("[#6]#[#7]")
        self.amidoxime_pattern = Chem.MolFromSmarts("[#6](=[#7])-[#7]-[#8]")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Conversion doesn't happen
        else:
            if self.condition_type == "bool":
                return 1  # Conversion occurs
            else:
                # Earlier conversion is generally better for this transformation
                return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction step converts a nitrile to an amidoxime.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check if any reactant has nitrile and any product has amidoxime
            has_nitrile_reactant = any(mol.HasSubstructMatch(self.nitrile_pattern) for mol in reactants)
            has_amidoxime_product = any(mol.HasSubstructMatch(self.amidoxime_pattern) for mol in products)
            
            # Additional check: ensure the nitrile is actually being converted
            # (not just present as a spectator)
            if has_nitrile_reactant and has_amidoxime_product:
                # Check that we're not just adding an amidoxime group to a molecule that already has one
                reactant_amidoximes = sum(len(mol.GetSubstructMatches(self.amidoxime_pattern)) for mol in reactants)
                product_amidoximes = sum(len(mol.GetSubstructMatches(self.amidoxime_pattern)) for mol in products)
                
                # True conversion should increase amidoxime count
                return product_amidoximes > reactant_amidoximes
            
            return False
            
        except Exception:
            return False
