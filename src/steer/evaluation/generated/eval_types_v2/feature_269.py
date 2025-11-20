"""Generated evaluation code for: Phthalimide protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PhthalimideProtectingGroup(BaseScoring):
    """
    Evaluates synthesis routes based on phthalimide protecting group strategy.
    Checks if phthalimide protection is used for amine groups and rewards
    early implementation of this protecting group strategy.
    """
    
    def __init__(self, config: Dict):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.functional_group = config["parameters"]["functional_group"] 
        self.cycles = config["parameters"]["cycles"]
        
        # SMARTS pattern for phthalimide group
        self.phthalimide_pattern = "O=C1c2ccccc2C(=O)N1"
        # SMARTS pattern for free amine
        self.amine_pattern = "[NH2,NH1]"
        
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score (0-10).
        Early use of phthalimide protection gets higher score.
        """
        if x < 0:
            return 0  # Protection strategy not found
        else:
            # Earlier protection is better - inverse relationship with depth
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves phthalimide protecting group formation.
        Returns True if phthalimide group appears in products but not in reactants.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    product_mols.append(mol)
            
            # Check for phthalimide protection reaction
            phthalimide_mol = Chem.MolFromSmarts(self.phthalimide_pattern)
            amine_mol = Chem.MolFromSmarts(self.amine_pattern)
            
            # Check if phthalimide appears in products
            phthalimide_in_products = any(
                mol.HasSubstructMatch(phthalimide_mol) for mol in product_mols
            )
            
            # Check if free amine exists in reactants
            amine_in_reactants = any(
                mol.HasSubstructMatch(amine_mol) for mol in reactant_mols
            )
            
            # Check if phthalimide reagent is present in reactants
            phthalimide_reagent_in_reactants = any(
                mol.HasSubstructMatch(phthalimide_mol) for mol in reactant_mols
            )
            
            # Protection reaction: amine + phthalimide reagent -> phthalimide-protected product
            return (phthalimide_in_products and 
                    amine_in_reactants and 
                    phthalimide_reagent_in_reactants)
                    
        except Exception:
            return False
