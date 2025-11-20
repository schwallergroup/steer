"""Generated evaluation code for: Late stage ether formation via methylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageEtherFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage ether formation via methylation.
    Detects Williamson ether synthesis forming C-O bonds with methylation patterns.
    """
    
    def __init__(self, config: Dict):
        self.bond_type = config["parameters"]["bond_formed"]  # "C-O"
        self.reaction_type = config["parameters"]["reaction_type"]  # "williamson_ether"
        self.substrate_pattern = config["parameters"]["substrate_pattern"]  # "[CH2]O"
        self.timing = config["parameters"]["timing"]  # "late"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ether formation doesn't happen
        else:
            # Late-stage formation is better, so higher depth fraction gives higher score
            return x * 10  # Scale to 0-10 range
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents late-stage ether formation via methylation.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1].split(".")
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants if r]
            
            if not prod_mol or not all(react_mols):
                return False
                
            # Check if product contains the methylated ether pattern
            ether_pattern = Chem.MolFromSmarts("[CH3]-O-[CH2]")  # Methyl ether pattern
            if not prod_mol.HasSubstructMatch(ether_pattern):
                return False
                
            # Check if reactants contain alcohol precursor and methylating agent
            alcohol_pattern = Chem.MolFromSmarts("[OH1]")  # Alcohol OH group
            methyl_source_pattern = Chem.MolFromSmarts("[CH3]-[Br,I,Cl]")  # Methylating agent
            
            has_alcohol = False
            has_methylating_agent = False
            
            for react_mol in react_mols:
                if react_mol.HasSubstructMatch(alcohol_pattern):
                    has_alcohol = True
                if react_mol.HasSubstructMatch(methyl_source_pattern):
                    has_methylating_agent = True
                    
            # Check for Williamson ether synthesis pattern
            return has_alcohol and has_methylating_agent
            
        except Exception:
            return False
