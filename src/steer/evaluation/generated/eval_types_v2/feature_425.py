"""Generated evaluation code for: Direct alcohol to chloride conversion attempt"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AlcoholToChlorideAttempt(BaseScoring):
    """
    Evaluates routes that attempt direct conversion of alcohol to chloride using 
    tosyl chloride without protecting competing amine nucleophiles.
    
    Checks for reactions where:
    1. Tosyl chloride (TsCl) is used as reagent
    2. Substrate contains both alcohol and amine groups
    3. This represents a problematic synthetic step due to competing nucleophiles
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
        
    def route_scoring(self, x) -> float:
        """
        Score the route based on depth of problematic reaction.
        Returns 0-10 where higher scores indicate better routes.
        """
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Penalize if condition is met
                return 0 if x >= 0 else 10
        else:
            if x < 0:  # Condition not met - good route
                return 10
            # Earlier occurrence is worse (less protected steps before)
            return max(0, 10 - (10 * (1 - x)))
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a problematic alcohol to chloride conversion.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Check for tosyl chloride in reactants
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Tosyl chloride pattern
            tosyl_chloride_pattern = Chem.MolFromSmarts("[CH3]-c1ccc(cc1)-S(=O)(=O)-Cl")
            has_tosyl_chloride = any(mol.HasSubstructMatch(tosyl_chloride_pattern) 
                                   for mol in reactant_mols if mol)
            
            if not has_tosyl_chloride:
                return False
            
            # Find the main substrate (largest non-reagent molecule)
            substrate_mol = None
            max_atoms = 0
            for mol in reactant_mols:
                if mol and mol.GetNumAtoms() > max_atoms:
                    # Skip if this looks like tosyl chloride
                    if not mol.HasSubstructMatch(tosyl_chloride_pattern):
                        substrate_mol = mol
                        max_atoms = mol.GetNumAtoms()
            
            if not substrate_mol:
                return False
            
            # Check if substrate contains both alcohol and amine
            alcohol_pattern = Chem.MolFromSmarts("[CH2,CH1,CH0]-[OH1]")  # Primary, secondary, tertiary alcohol
            amine_pattern = Chem.MolFromSmarts("[NH2,NH1,NH0]")  # Primary, secondary, tertiary amine
            
            has_alcohol = substrate_mol.HasSubstructMatch(alcohol_pattern)
            has_amine = substrate_mol.HasSubstructMatch(amine_pattern)
            
            # Check if conversion involves alcohol to chloride
            product_mol = Chem.MolFromSmiles(products_smiles)
            if not product_mol:
                return False
                
            chloride_pattern = Chem.MolFromSmarts("[CH2,CH1,CH0]-Cl")  # Alkyl chloride
            has_chloride_product = product_mol.HasSubstructMatch(chloride_pattern)
            
            return has_alcohol and has_amine and has_chloride_product
            
        except Exception:
            return False
