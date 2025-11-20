"""Generated evaluation code for: Late stage aryl chloride dechlorination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageArylChlorideDechlorination(BaseScoring):
    """
    Evaluates synthesis routes for late-stage aryl chloride dechlorination reactions.
    Checks if an aryl chloride is converted to an unsubstituted aromatic compound
    in the final synthetic steps.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config["parameters"]["step_position"]
        self.total_steps = config["parameters"]["total_steps"]
        # Calculate target depth as fraction
        self.target_depth = self.step_position / self.total_steps
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        # Score higher for reactions closer to the target late-stage position
        penalty = abs(x - self.target_depth)
        return max(0, 1 - penalty * 5)  # Scale penalty to 0-1 range
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction is an aryl chloride dechlorination"""
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            rxn_parts = rxn_smiles.split(">>")
            
            if len(rxn_parts) != 2:
                return False
                
            reactant_smiles = rxn_parts[0]
            product_smiles = rxn_parts[1]
            
            # Parse molecules
            reactant = Chem.MolFromSmiles(reactant_smiles)
            product = Chem.MolFromSmiles(product_smiles)
            
            if reactant is None or product is None:
                return False
            
            # Check if reactant contains aryl chloride
            aryl_chloride_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[Cl:2]")
            if not reactant.HasSubstructMatch(aryl_chloride_pattern):
                return False
            
            # Get atom mapping for the chlorinated carbon
            matches = reactant.GetSubstructMatches(aryl_chloride_pattern)
            if not matches:
                return False
            
            # Check if the chlorine is removed in the product
            for match in matches:
                carbon_idx, chlorine_idx = match
                carbon_atom = reactant.GetAtomWithIdx(carbon_idx)
                chlorine_atom = reactant.GetAtomWithIdx(chlorine_idx)
                
                carbon_map_num = carbon_atom.GetAtomMapNum()
                chlorine_map_num = chlorine_atom.GetAtomMapNum()
                
                if carbon_map_num == 0:
                    continue
                
                # Find corresponding carbon in product
                product_carbon = None
                for atom in product.GetAtoms():
                    if atom.GetAtomMapNum() == carbon_map_num:
                        product_carbon = atom
                        break
                
                if product_carbon is None:
                    continue
                
                # Check if chlorine is no longer bound to this carbon
                product_has_chlorine = False
                for neighbor in product_carbon.GetNeighbors():
                    if neighbor.GetSymbol() == "Cl":
                        product_has_chlorine = True
                        break
                
                # Also verify chlorine atom is not in product (if mapped)
                chlorine_in_product = False
                if chlorine_map_num > 0:
                    for atom in product.GetAtoms():
                        if atom.GetAtomMapNum() == chlorine_map_num:
                            chlorine_in_product = True
                            break
                
                # Dechlorination occurred if chlorine is no longer bound and not in product
                if not product_has_chlorine and not chlorine_in_product:
                    return True
            
            return False
            
        except Exception:
            return False
