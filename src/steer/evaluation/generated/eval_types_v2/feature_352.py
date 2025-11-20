"""Generated evaluation code for: Benzophenone imine protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzophenoneImineProtection(BaseScoring):
    """
    Evaluates synthesis routes for the use of benzophenone imine as a protecting group
    for primary amines. Returns higher scores when this protection strategy is used
    earlier in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.0)
        
        # SMARTS patterns for benzophenone imine protection/deprotection
        self.benzophenone_imine_pattern = "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#7]-[#6])-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2"
        self.primary_amine_pattern = "[#6]-[#7H2]"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Protection strategy not used
        
        if self.condition_type == "bool":
            return 1  # Strategy is present
        else:
            # Earlier use of protection (lower depth fraction) gets higher score
            return 1 - x
    
    def hit_condition(self, d):
        """
        Check if this reaction involves benzophenone imine protection of a primary amine
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            products = rxn_parts[0]
            reactants = rxn_parts[1]
            
            prod_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".") if r.strip()]
            
            if not prod_mol or not all(reactant_mols):
                return False
            
            # Check if product contains benzophenone imine
            imine_pattern = Chem.MolFromSmarts(self.benzophenone_imine_pattern)
            if not prod_mol.HasSubstructMatch(imine_pattern):
                return False
            
            # Check if any reactant has a primary amine that gets protected
            amine_pattern = Chem.MolFromSmarts(self.primary_amine_pattern)
            has_primary_amine_reactant = any(mol.HasSubstructMatch(amine_pattern) for mol in reactant_mols)
            
            # Check if benzophenone (Ph2CO) is a reactant
            benzophenone_pattern = Chem.MolFromSmarts("[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1-[#6](=[#8])-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2")
            has_benzophenone_reactant = any(mol.HasSubstructMatch(benzophenone_pattern) for mol in reactant_mols)
            
            return has_primary_amine_reactant and has_benzophenone_reactant
            
        except Exception:
            return False
