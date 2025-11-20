"""Generated evaluation code for: Grignard addition to benzophenone for trityl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class GrignardBenzophenoneAddition(BaseScoring):
    """
    Evaluates synthesis routes for Grignard addition to benzophenone to form trityl compounds.
    Detects the formation of triphenylmethanol or similar tertiary alcohols from benzophenone
    and phenyl Grignard reagents.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.5)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Earlier occurrence is better for this classical reaction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction represents Grignard addition to benzophenone.
        Looks for benzophenone -> triphenylmethanol transformation.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = rxn[0].split(".")
            products = rxn[1].split(".")
            
            # Define SMARTS patterns
            benzophenone_pattern = "c1ccccc1C(=O)c2ccccc2"  # Benzophenone
            grignard_pattern = "[Mg][Br,Cl,I]"  # Grignard reagent
            phenyl_grignard_pattern = "c1ccccc1[Mg][Br,Cl,I]"  # Phenyl Grignard
            triphenylmethanol_pattern = "c1ccccc1C(O)(c2ccccc2)c3ccccc3"  # Triphenylmethanol
            tertiary_alcohol_pattern = "C(O)(c1ccccc1)(c2ccccc2)c3ccccc3"  # Tertiary alcohol with three phenyls
            
            # Convert to molecule objects
            reactant_mols = []
            for r_smiles in reactants:
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smiles in products:
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    product_mols.append(mol)
            
            # Check for benzophenone in reactants
            has_benzophenone = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(benzophenone_pattern))
                for mol in reactant_mols
            )
            
            # Check for Grignard reagent in reactants
            has_grignard = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(grignard_pattern))
                for mol in reactant_mols
            )
            
            # Check for phenyl Grignard specifically
            has_phenyl_grignard = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(phenyl_grignard_pattern))
                for mol in reactant_mols
            )
            
            # Check for triphenylmethanol or similar tertiary alcohol in products
            has_trityl_product = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(triphenylmethanol_pattern)) or
                mol.HasSubstructMatch(Chem.MolFromSmarts(tertiary_alcohol_pattern))
                for mol in product_mols
            )
            
            # Reaction is hit if we have benzophenone + Grignard -> tertiary alcohol
            return has_benzophenone and (has_grignard or has_phenyl_grignard) and has_trityl_product
            
        except (KeyError, AttributeError, ValueError):
            return False
