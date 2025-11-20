"""Generated evaluation code for: Late stage Williamson ether formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageWilliamsonEther(BaseScoring):
    """
    Evaluates synthesis routes for late-stage Williamson ether formation.
    
    Detects when a Williamson ether synthesis reaction occurs and scores
    based on how late in the synthesis it appears (later is better).
    """
    
    def __init__(self, config: Dict):
        self.target_step_position = config.get("parameters", {}).get("step_position", 1)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether formation doesn't occur
        else:
            # Late-stage is better, so invert the depth fraction
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by checking for:
        1. Formation of C-O-C ether bond
        2. Presence of alkoxide nucleophile pattern
        3. Alkyl halide or tosylate electrophile pattern
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants, products = rxn_smiles.split(">>")
            
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products)
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".") if r.strip()]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check for ether formation in product
            ether_pattern = Chem.MolFromSmarts("[C,c]-O-[C,c]")
            if not product_mol.HasSubstructMatch(ether_pattern):
                return False
            
            # Look for characteristic Williamson ether reactants
            has_alkoxide = False
            has_alkyl_halide = False
            
            for reactant in reactant_mols:
                if not reactant:
                    continue
                    
                # Check for alkoxide nucleophile (phenoxide or alkoxide)
                alkoxide_patterns = [
                    "[O-]-[C,c]",  # General alkoxide/phenoxide
                    "[OH]-[c]",    # Phenol (can be deprotonated)
                    "[OH]-[C]"     # Alcohol (can be deprotonated)
                ]
                
                for pattern in alkoxide_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alkoxide = True
                        break
                
                # Check for alkyl halide or tosylate electrophile
                electrophile_patterns = [
                    "[C]-[Cl,Br,I]",           # Alkyl halide
                    "[C]-[#16](=O)(=O)-[c]",   # Tosylate/mesylate
                    "[c]-[Cl,Br,I,F]"          # Aryl halide (for SNAr)
                ]
                
                for pattern in electrophile_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alkyl_halide = True
                        break
            
            return has_alkoxide and has_alkyl_halide
            
        except Exception:
            return False
