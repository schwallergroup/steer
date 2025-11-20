"""Generated evaluation code for: Late stage reductive amination coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageReductiveAmination(BaseScoring):
    """
    Evaluates whether reductive amination occurs in the final step of the synthesis route.
    Rewards routes where the final reaction is a reductive amination coupling of two complex fragments.
    """
    
    def __init__(self, config: Dict):
        self.target_timing = config.get("timing", "final_step")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reductive amination doesn't happen
        elif self.target_timing == "final_step":
            # For final step, we want x to be very close to 1 (latest possible)
            if x > 0.9:
                return 10  # Perfect score for final step
            else:
                return x * 10  # Scale to 0-10, favoring later steps
        else:
            # For other timing preferences, could be extended
            return 1 - abs(x - 0.5) * 2  # Favor middle timing
    
    def hit_condition(self, d) -> bool:
        """
        Detects reductive amination by looking for:
        1. Formation of C-N bond between amine and carbonyl carbon
        2. Reduction of intermediate imine/iminium to amine
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Look for reductive amination patterns
            return self._is_reductive_amination(reactants, product)
            
        except Exception:
            return False
    
    def _is_reductive_amination(self, reactants, product) -> bool:
        """
        Identifies reductive amination by checking:
        1. Presence of carbonyl compound (aldehyde/ketone) in reactants
        2. Presence of amine in reactants  
        3. Formation of new C-N bond in product
        4. Typical reducing agents (NaBH4, NaCNBH3, etc.)
        """
        # Common carbonyl patterns (aldehydes and ketones)
        carbonyl_patterns = [
            "[CH1]=O",  # Aldehyde
            "[CH2][CH1]=O",  # Aldehyde with carbon
            "[C]([H])=O",  # Aldehyde general
            "[C](=O)[C]",  # Ketone
            "[C]=O"  # General carbonyl
        ]
        
        # Amine patterns
        amine_patterns = [
            "[NH2]",  # Primary amine
            "[NH1]",  # Secondary amine
            "[N]([H])[H]",  # Primary amine explicit
            "[N]([H])[C]",  # Secondary amine
        ]
        
        # Common reducing agents for reductive amination
        reducing_agents = [
            "B",  # Borohydrides
            "[BH4-]",  # Sodium borohydride
            "[BH3]",  # Borane
            "B([H])",  # Borane derivatives
        ]
        
        has_carbonyl = False
        has_amine = False
        has_reducing_agent = False
        
        # Check reactants for required components
        for reactant in reactants:
            # Check for carbonyl
            for pattern in carbonyl_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_carbonyl = True
                        break
                except:
                    continue
            
            # Check for amine
            for pattern in amine_patterns:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_amine = True
                        break
                except:
                    continue
            
            # Check for reducing agent
            for pattern in reducing_agents:
                try:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_reducing_agent = True
                        break
                except:
                    continue
        
        # Also check if product has characteristic secondary/tertiary amine
        product_amine_patterns = [
            "[N]([C])[C]",  # Secondary amine (C-N-C)
            "[N]([C])([C])[C]",  # Tertiary amine
        ]
        
        has_product_amine = False
        for pattern in product_amine_patterns:
            try:
                if product.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    has_product_amine = True
                    break
            except:
                continue
        
        # Reductive amination requires carbonyl + amine reactants + reducing conditions + amine product
        return has_carbonyl and has_amine and (has_reducing_agent or has_product_amine)
