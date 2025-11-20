"""Generated evaluation code for: Triflate intermediate for carbonylation coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class TriflateIntermediateCarbonylation(BaseScoring):
    """
    Evaluates synthesis routes for the formation of triflate intermediates from ketones
    followed by carbonylation coupling reactions. Checks if a C=O bond is converted
    to an enol triflate intermediate for subsequent Pd-catalyzed carbonylation.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.3)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Transformation doesn't occur
        else:
            # Earlier triflate formation is generally better for synthetic planning
            return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if the reaction involves ketone to triflate conversion followed by carbonylation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for triflate formation or carbonylation with triflate
            return self._has_triflate_carbonylation_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _has_triflate_carbonylation_pattern(self, reactants, products) -> bool:
        """
        Detects triflate intermediate formation from ketones or carbonylation of triflates.
        """
        # Triflate group pattern (OTf)
        triflate_pattern = Chem.MolFromSmarts("[O:1]S(=O)(=O)C(F)(F)F")
        
        # Ketone pattern
        ketone_pattern = Chem.MolFromSmarts("[C:1](=O)[C,c]")
        
        # Ester pattern (product of carbonylation)
        ester_pattern = Chem.MolFromSmarts("[C:1](=O)O[C,c]")
        
        if not all([triflate_pattern, ketone_pattern, ester_pattern]):
            return False
        
        # Check for ketone to triflate conversion
        has_ketone_reactant = any(mol.HasSubstructMatch(ketone_pattern) for mol in reactants)
        has_triflate_product = any(mol.HasSubstructMatch(triflate_pattern) for mol in products)
        
        if has_ketone_reactant and has_triflate_product:
            return True
        
        # Check for triflate carbonylation (triflate reactant -> ester product)
        has_triflate_reactant = any(mol.HasSubstructMatch(triflate_pattern) for mol in reactants)
        has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in products)
        has_co_reactant = any("C#O" in Chem.MolToSmiles(mol) or 
                             "[C-]#[O+]" in Chem.MolToSmiles(mol) for mol in reactants)
        
        if has_triflate_reactant and has_ester_product and has_co_reactant:
            return True
        
        return False
