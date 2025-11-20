"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki coupling reaction for biaryl formation occurs late in the synthesis.
    
    Detects Suzuki cross-coupling reactions that form C-C bonds between aromatic rings
    and rewards reactions that occur closer to the final product (late stage).
    """
    
    def __init__(self, config: Dict):
        pass  # No configuration needed for this feature
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        
        Args:
            x: Depth fraction where Suzuki coupling occurs (-1 if not found)
            
        Returns:
            Score from 0-1, where 1 is best (late stage coupling)
        """
        if x < 0:
            return 0  # No Suzuki coupling found
        else:
            return 1 - x  # Late-stage (low depth fraction) is better
    
    def hit_condition(self, d) -> bool:
        """
        Check if a reaction node represents a Suzuki coupling for biaryl formation.
        
        Args:
            d: Reaction node dictionary containing metadata
            
        Returns:
            True if this is a Suzuki coupling forming a biaryl bond
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check if this is a Suzuki-type reaction (has boron and halogen/pseudo-halogen)
            has_boron_reactant = False
            has_halogen_reactant = False
            
            for reactant in reactants:
                # Check for boronic acid/ester patterns
                boron_patterns = [
                    "[B]([OH])[OH]",  # Boronic acid
                    "[B]1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
                    "[B](OC)OC"  # Methyl boronate
                ]
                
                for pattern in boron_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_boron_reactant = True
                        break
                
                # Check for halogen/triflate on aromatic carbon
                halogen_patterns = [
                    "c[Cl]",  # Aryl chloride
                    "c[Br]",  # Aryl bromide  
                    "c[I]",   # Aryl iodide
                    "cOS(=O)(=O)C(F)(F)F"  # Aryl triflate
                ]
                
                for pattern in halogen_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_halogen_reactant = True
                        break
            
            if not (has_boron_reactant and has_halogen_reactant):
                return False
            
            # Check if a biaryl bond was formed
            return self._detects_biaryl_formation(reactants, product)
            
        except Exception:
            return False
    
    def _detects_biaryl_formation(self, reactants, product) -> bool:
        """
        Check if the reaction formed a new C-C bond between two aromatic rings.
        
        Args:
            reactants: List of reactant molecules
            product: Product molecule
            
        Returns:
            True if a biaryl C-C bond was formed
        """
        # Count biaryl bonds in reactants vs product
        biaryl_pattern = Chem.MolFromSmarts("c-c")  # Aromatic C-C bond
        
        reactant_biaryl_count = sum(len(mol.GetSubstructMatches(biaryl_pattern)) 
                                   for mol in reactants)
        product_biaryl_count = len(product.GetSubstructMatches(biaryl_pattern))
        
        # If product has more biaryl bonds than sum of reactants, a new one was formed
        return product_biaryl_count > reactant_biaryl_count
