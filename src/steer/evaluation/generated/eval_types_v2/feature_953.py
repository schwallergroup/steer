"""Generated evaluation code for: Late stage ester saponification"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEsterSaponification(BaseScoring):
    """
    Evaluates whether ester saponification (hydrolysis of ethyl ester to carboxylic acid) 
    occurs at a late stage in the synthesis route. Higher scores indicate the reaction 
    occurs closer to the final step.
    """
    
    def __init__(self, config: Dict):
        self.stage_preference = config.get("stage", "final")  # "final" means prefer late stage
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Saponification doesn't happen
        else:
            # Late-stage saponification is better (closer to 1.0 depth fraction)
            # Score from 0-10, with 10 being latest stage
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents ester saponification
        (ethyl ester -> carboxylic acid conversion)
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1].split(".")
        
        try:
            # Parse reactant molecule
            reactant_mol = Chem.MolFromSmiles(reactants)
            if reactant_mol is None:
                return False
                
            # Check for ethyl ester pattern in reactant
            ethyl_ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][CH2:4][CH3:5]")
            if not reactant_mol.HasSubstructMatch(ethyl_ester_pattern):
                return False
                
            # Check that products contain carboxylic acid
            carboxylic_acid_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[OH:3]")
            
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue
                    
                if product_mol.HasSubstructMatch(carboxylic_acid_pattern):
                    # Verify the carbon atoms match (same atom mapping)
                    reactant_matches = reactant_mol.GetSubstructMatches(ethyl_ester_pattern)
                    product_matches = product_mol.GetSubstructMatches(carboxylic_acid_pattern)
                    
                    for r_match in reactant_matches:
                        r_carbon_map = reactant_mol.GetAtomWithIdx(r_match[0]).GetAtomMapNum()
                        if r_carbon_map == 0:
                            continue
                            
                        for p_match in product_matches:
                            p_carbon_map = product_mol.GetAtomWithIdx(p_match[0]).GetAtomMapNum()
                            if r_carbon_map == p_carbon_map:
                                return True
                                
        except Exception:
            return False
            
        return False
