"""Generated evaluation code for: Late stage amide coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmideCoupling(BaseScoring):
    """
    Evaluates whether amide coupling occurs in the late stages of synthesis.
    Detects amide bond formation reactions and penalizes if they occur too early.
    """
    
    def __init__(self, config: Dict):
        self.step_threshold = config.get("step_threshold", 3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # No amide coupling found
        # Late-stage coupling is better - penalize early coupling
        # x is depth fraction (0 = root, 1 = leaves)
        if x > 0.7:  # Very late stage
            return 1.0
        elif x > 0.5:  # Moderately late
            return 0.7
        else:  # Too early
            return 0.3
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves amide bond formation"""
        try:
            mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            prod_smiles, react_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(prod_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in react_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if amide bond is formed
            return self._is_amide_formation(product, reactants)
            
        except Exception:
            return False
    
    def _is_amide_formation(self, product, reactants) -> bool:
        """
        Detect amide bond formation by checking for:
        1. Amide bond in product that wasn't in reactants
        2. Presence of carboxylic acid/ester and amine in reactants
        """
        # Define patterns
        amide_pattern = Chem.MolFromSmarts("[C](=O)[NH,NH2]")
        carboxylic_acid = Chem.MolFromSmarts("[C](=O)[OH]")
        ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
        amine_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        
        if not amide_pattern:
            return False
        
        # Check if product has amide bond
        has_amide_in_product = product.HasSubstructMatch(amide_pattern)
        if not has_amide_in_product:
            return False
        
        # Check if any reactant has both carboxylic acid/ester and another has amine
        has_carboxyl = False
        has_amine = False
        
        for reactant in reactants:
            if carboxylic_acid and reactant.HasSubstructMatch(carboxylic_acid):
                has_carboxyl = True
            if ester_pattern and reactant.HasSubstructMatch(ester_pattern):
                has_carboxyl = True
            if amine_pattern and reactant.HasSubstructMatch(amine_pattern):
                has_amine = True
        
        # Alternative: check for coupling reagents (EDC, HATU, etc.)
        coupling_reagents = [
            "CCN=C=NCCCN(C)C",  # EDC
            "CN(C)C(=O)C1=CN=CC=C1",  # DMAP-like
            "O=C1OC(=O)C(C(F)(F)F)=C1C(F)(F)F"  # Anhydride coupling
        ]
        
        has_coupling_reagent = False
        for reactant in reactants:
            react_smiles = Chem.MolToSmiles(reactant)
            for reagent_pattern in coupling_reagents:
                try:
                    reagent_mol = Chem.MolFromSmiles(reagent_pattern)
                    if reagent_mol and reactant.HasSubstructMatch(reagent_mol):
                        has_coupling_reagent = True
                        break
                except:
                    continue
        
        return has_amide_in_product and (has_carboxyl and has_amine) or has_coupling_reagent
