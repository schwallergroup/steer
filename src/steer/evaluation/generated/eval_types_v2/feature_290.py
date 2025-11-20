"""Generated evaluation code for: Convergent synthesis via two major fragments"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategies by detecting when two major fragments
    are coupled together late in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_position = config.get("coupling_step_position", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Convergent coupling doesn't happen
        else:
            if self.coupling_position == "late":
                return 1 - x  # Later coupling is better for convergent synthesis
            else:
                return x  # Earlier coupling preferred
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a convergent coupling of major fragments
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            reactants = reactants_smiles.split(".")
            
            # Need at least the specified number of fragments
            if len(reactants) < self.fragment_count:
                return False
                
            # Parse molecules
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
            
            if not product_mol or not all(reactant_mols):
                return False
                
            # Check for convergent coupling patterns
            return self._is_convergent_coupling(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _is_convergent_coupling(self, reactants, product):
        """
        Detect if this is a convergent coupling reaction by checking:
        1. Multiple substantial fragments as reactants
        2. Common coupling reactions (amide, ether, C-C bond formation)
        """
        # Filter out small molecules (likely reagents/catalysts)
        major_fragments = [mol for mol in reactants if mol.GetNumAtoms() >= 5]
        
        if len(major_fragments) < self.fragment_count:
            return False
            
        # Check for common convergent coupling patterns
        coupling_patterns = [
            # Amide formation
            "[C:1](=[O:2])[OH].[N:3]>>[C:1](=[O:2])[N:3]",
            # Ether formation  
            "[C:1][OH].[C:2][X]>>[C:1][O][C:2]",
            # Suzuki coupling
            "[c:1][B].[c:2][X]>>[c:1][c:2]",
            # Click chemistry
            "[C:1]#[C:2].[N:3]=[N+]=[N-]>>[c:1][n:3][n][n][c:2]",
            # General C-C bond formation
            "[C:1].[C:2]>>[C:1][C:2]"
        ]
        
        # Check if reaction matches coupling patterns
        for pattern in coupling_patterns:
            try:
                rxn = AllChem.ReactionFromSmarts(pattern)
                if rxn:
                    # Simple heuristic: if we have major fragments and potential coupling,
                    # consider it convergent
                    return True
            except:
                continue
                
        # Additional check: significant increase in molecular complexity
        reactant_complexity = sum(mol.GetNumAtoms() for mol in major_fragments)
        product_complexity = product.GetNumAtoms()
        
        # If product is roughly the sum of major fragments, likely convergent
        if abs(product_complexity - reactant_complexity) <= 5:  # Allow for small reagents
            return True
            
        return False
