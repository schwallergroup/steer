"""Generated evaluation code for: Symmetric dihalide alkylation approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SymmetricDihalideAlkylation(BaseScoring):
    """
    Evaluates synthesis routes for the presence of symmetric dihalide alkylation reactions.
    Specifically detects alkylation using 1,4-dichlorobutane or similar symmetric dihalides
    that can lead to dimerization side reactions.
    """
    
    def __init__(self, config: Dict):
        self.reaction_smarts = config.get("reaction_smarts", "[C-:1].[Cl:2][CH2:3][CH2:4][CH2:5][CH2:6][Cl:7]>>[C:1][CH2:3][CH2:4][CH2:5][CH2:6][Cl:7]")
        self.reagent_pattern = config.get("reagent_pattern", "ClCCCCCl")
        
        # Compile the reaction pattern
        self.rxn_pattern = AllChem.ReactionFromSmarts(self.reaction_smarts)
        self.reagent_mol = Chem.MolFromSmarts(self.reagent_pattern)
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth where symmetric dihalide alkylation occurs.
        Earlier occurrence (lower depth fraction) gives higher penalty.
        """
        if x < 0:
            return 0  # Reaction not found - neutral score
        else:
            return x * 10  # Higher penalty for earlier use (problematic side reactions)
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node contains symmetric dihalide alkylation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles")
        
        if not mapped_rxn:
            return False
            
        try:
            # Split reaction SMILES
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check for symmetric dihalide reagent pattern
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol and reactant_mol.HasSubstructMatch(self.reagent_mol):
                    return True
            
            # Alternative check: look for the reaction pattern match
            try:
                rxn_mol = AllChem.ReactionFromSmarts(mapped_rxn)
                if rxn_mol:
                    # Check if reaction matches the symmetric dihalide alkylation pattern
                    return self._matches_alkylation_pattern(reactants, products)
            except:
                pass
                
        except Exception:
            pass
            
        return False
    
    def _matches_alkylation_pattern(self, reactants, products):
        """
        Helper method to identify alkylation with symmetric dihalide.
        Looks for pattern where one chlorine is consumed and chain is extended.
        """
        try:
            # Look for dihalide pattern in reactants
            dihalide_found = False
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    # Check for symmetric dihalide (ClCnCl pattern)
                    dihalide_patterns = [
                        Chem.MolFromSmarts("ClCCCCCl"),  # 1,4-dichlorobutane
                        Chem.MolFromSmarts("ClCCCl"),    # 1,3-dichloropropane
                        Chem.MolFromSmarts("ClCCCCCCl"), # 1,5-dichloropentane
                        Chem.MolFromSmarts("ClC[CH2]CCl") # generic dihalide
                    ]
                    
                    for pattern in dihalide_patterns:
                        if pattern and mol.HasSubstructMatch(pattern):
                            dihalide_found = True
                            break
                    
                    if dihalide_found:
                        break
            
            return dihalide_found
            
        except Exception:
            return False
