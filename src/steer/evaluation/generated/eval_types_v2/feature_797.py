"""Generated evaluation code for: Sequential SNAr reactions on pyridine"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialSNArPyridine(MultiRxnCondBase):
    """
    Evaluates synthesis routes for sequential nucleophilic aromatic substitution (SNAr) reactions on pyridine rings.
    Checks for the presence of multiple SNAr reactions involving pyridine substrates.
    """
    
    def __init__(self, config):
        self.target_count = config.get("count", 3)
        self.substrate = config.get("substrate", "pyridine")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        snar_count = sum(1 for r in reactions if self.detect_snar_pyridine(r))
        
        condition = snar_count >= self.target_count
        return condition, len(reactions)
    
    def detect_snar_pyridine(self, rxn):
        """
        Detects SNAr reactions on pyridine by checking for:
        1. Pyridine ring in reactants
        2. Nucleophilic substitution pattern (leaving group replacement)
        3. Aromatic nitrogen activation
        """
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check for pyridine ring in reactants
            pyridine_pattern = Chem.MolFromSmarts("c1ccncc1")  # pyridine ring
            has_pyridine = False
            
            for reactant_smiles in reactants:
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant and reactant.HasSubstructMatch(pyridine_pattern):
                    has_pyridine = True
                    break
            
            if not has_pyridine:
                return False
            
            # Check for typical SNAr leaving groups being replaced
            leaving_groups = [
                Chem.MolFromSmarts("c-F"),    # fluoride
                Chem.MolFromSmarts("c-Cl"),   # chloride  
                Chem.MolFromSmarts("c-Br"),   # bromide
                Chem.MolFromSmarts("c-I"),    # iodide
                Chem.MolFromSmarts("c-[N+](=O)[O-]"), # nitro group
            ]
            
            # Check if reactant has leaving group that's absent in product
            reactant_has_lg = False
            product_lacks_lg = True
            
            for reactant_smiles in reactants:
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant and reactant.HasSubstructMatch(pyridine_pattern):
                    # Check if this pyridine has a leaving group
                    for lg_pattern in leaving_groups:
                        if reactant.HasSubstructMatch(lg_pattern):
                            reactant_has_lg = True
                            
                            # Check if products lack this leaving group on pyridine
                            for product_smiles in products:
                                product = Chem.MolFromSmiles(product_smiles)
                                if (product and 
                                    product.HasSubstructMatch(pyridine_pattern) and 
                                    product.HasSubstructMatch(lg_pattern)):
                                    product_lacks_lg = False
                                    break
                            break
                    break
            
            # SNAr detected if pyridine with leaving group becomes pyridine without it
            return has_pyridine and reactant_has_lg and product_lacks_lg
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Score based on whether the target count of SNAr reactions is achieved.
        x is the fraction of reactions that are SNAr on pyridine.
        """
        if x < 0:
            return 0  # Condition not met
        
        # Calculate actual count from fraction and total reactions
        actual_count = x * self.get_total_reactions()
        
        if actual_count >= self.target_count:
            return 10  # Perfect score for meeting target
        else:
            # Partial score based on how close we are to target
            return (actual_count / self.target_count) * 10
