"""Generated evaluation code for: Convergent synthesis via two fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two independently prepared fragments
    are coupled at a specified stage (early/late) in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.coupling_stage = config["parameters"]["coupling_stage"]  # "early" or "late"
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        
        if self.coupling_stage == "late":
            return 1 - x  # Later coupling is better (closer to 1.0)
        elif self.coupling_stage == "early":
            return x  # Earlier coupling is better (closer to 0.0)
        else:
            return 0.5 if x >= 0 else 0  # Any convergent coupling is acceptable
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of the specified
        number of fragments by analyzing the reaction stoichiometry.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            if len(rxn) != 2:
                return False
                
            product_smiles = rxn[0]
            reactant_smiles = rxn[1]
            
            # Count number of reactant molecules (fragments)
            reactants = reactant_smiles.split(".")
            
            # Filter out small molecules (catalysts, reagents) by molecular weight
            significant_reactants = []
            for reactant in reactants:
                mol = Chem.MolFromSmiles(reactant)
                if mol is not None:
                    mw = Chem.Descriptors.MolWt(mol)
                    # Consider molecules with MW > 100 as significant fragments
                    if mw > 100:
                        significant_reactants.append(reactant)
            
            # Check if we have the expected number of significant fragments
            if len(significant_reactants) != self.fragment_count:
                return False
            
            # Verify this is actually a coupling reaction (C-C, C-N, C-O bond formation)
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
            
            # Check for common coupling reaction patterns
            coupling_patterns = [
                "[C:1]-[C:2]",  # C-C coupling
                "[C:1]-[N:2]",  # C-N coupling  
                "[C:1]-[O:2]",  # C-O coupling
                "[C:1]=[C:2]",  # Alkene formation
                "[c:1]-[c:2]",  # Aromatic coupling
                "[C:1]#[C:2]",  # Alkyne formation
            ]
            
            # At least one coupling pattern should be present
            for pattern in coupling_patterns:
                if product_mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
            
            return False
            
        except Exception:
            return False
