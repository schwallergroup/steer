"""Generated evaluation code for: Gabriel synthesis amine protection strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GabrielSynthesis(BaseScoring):
    """
    Evaluates routes that use Gabriel synthesis amine protection strategy.
    Checks for the presence of phthalimide protection followed by hydrazine deprotection.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
        
        # Phthalimide protecting group pattern
        self.phthalimide_pattern = Chem.MolFromSmarts("C1=CC=C2C(=C1)C(=O)N(C2=O)[CH2]")
        # Hydrazine deprotection pattern
        self.hydrazine_pattern = Chem.MolFromSmarts("NN")
        
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            return 1.0 if x >= 0 else 0.0
        else:
            if x < 0:
                return 0.0
            # Earlier use of Gabriel synthesis is preferred (lower depth)
            return max(0.0, 10.0 * (1.0 - abs(x - self.target_depth)))
    
    def hit_condition(self, d):
        """
        Check if this reaction involves Gabriel synthesis protection/deprotection.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for phthalimide protection step
            # Product has phthalimide, reactant doesn't
            has_phthalimide_product = any(mol.HasSubstructMatch(self.phthalimide_pattern) for mol in products)
            has_phthalimide_reactant = any(mol.HasSubstructMatch(self.phthalimide_pattern) for mol in reactants)
            
            if has_phthalimide_product and not has_phthalimide_reactant:
                return True
            
            # Check for Gabriel deprotection step
            # Reactant has phthalimide, products include primary amine and hydrazine is used
            if has_phthalimide_reactant and not has_phthalimide_product:
                # Check if hydrazine is used as reagent
                has_hydrazine = any(mol.HasSubstructMatch(self.hydrazine_pattern) for mol in reactants)
                
                # Check if primary amine is formed
                primary_amine_pattern = Chem.MolFromSmarts("[NH2][CH2]")
                has_primary_amine_product = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in products)
                
                if has_hydrazine and has_primary_amine_product:
                    return True
            
            return False
            
        except Exception:
            return False
