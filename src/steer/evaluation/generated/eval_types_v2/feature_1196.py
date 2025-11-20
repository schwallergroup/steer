"""Generated evaluation code for: Late stage lactone to lactol reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LactoneToLactolReduction(BaseScoring):
    """
    Evaluates synthesis routes for late-stage lactone to lactol reduction reactions.
    
    This scoring function identifies routes where a lactone (cyclic ester) is reduced
    to a lactol (cyclic hemiacetal) in the later stages of synthesis, particularly
    as a final transformation step.
    """
    
    def __init__(self, config: Dict):
        self.timing_preference = config.get("timing", "late")
        self.step_position = config.get("step_position", "final")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reduction doesn't happen
        
        # For late-stage preference, higher depth fraction is better
        if self.timing_preference == "late":
            if self.step_position == "final" and x >= 0.9:
                return 10  # Bonus for final step reduction
            return x * 10  # Later is better
        else:
            return (1 - x) * 10  # Earlier is better
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves lactone to lactol reduction.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants = rxn[0]
            products = rxn[1]
            
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Define lactone pattern (cyclic ester)
            # Match 5, 6, or 7-membered lactones
            lactone_patterns = [
                "[#6]1[#6][#6][#6]C(=O)O1",  # 5-membered (gamma-lactone)
                "[#6]1[#6][#6][#6][#6]C(=O)O1",  # 6-membered (delta-lactone)
                "[#6]1[#6][#6][#6][#6][#6]C(=O)O1"  # 7-membered (epsilon-lactone)
            ]
            
            # Define lactol pattern (cyclic hemiacetal)
            # C-OH with ring oxygen
            lactol_patterns = [
                "[#6]1[#6][#6][#6]C(O)O1",  # 5-membered lactol
                "[#6]1[#6][#6][#6][#6]C(O)O1",  # 6-membered lactol
                "[#6]1[#6][#6][#6][#6][#6]C(O)O1"  # 7-membered lactol
            ]
            
            # Check for lactone in reactants
            has_lactone_reactant = False
            for mol in reactant_mols:
                for pattern in lactone_patterns:
                    lactone_smarts = Chem.MolFromSmarts(pattern)
                    if lactone_smarts and mol.HasSubstructMatch(lactone_smarts):
                        has_lactone_reactant = True
                        break
                if has_lactone_reactant:
                    break
            
            # Check for lactol in products
            has_lactol_product = False
            for mol in product_mols:
                for pattern in lactol_patterns:
                    lactol_smarts = Chem.MolFromSmarts(pattern)
                    if lactol_smarts and mol.HasSubstructMatch(lactol_smarts):
                        has_lactol_product = True
                        break
                if has_lactol_product:
                    break
            
            # Also check for presence of reducing agents (optional additional validation)
            reducing_agents = [
                "[#1][#1]",  # H2
                "B([#1])([#1])[#1]",  # BH3
                "[Li][Al]",  # LiAlH4 components
                "NaBH4"  # Sodium borohydride
            ]
            
            has_reducing_agent = False
            for mol in reactant_mols:
                for agent in reducing_agents:
                    try:
                        agent_smarts = Chem.MolFromSmarts(agent)
                        if agent_smarts and mol.HasSubstructMatch(agent_smarts):
                            has_reducing_agent = True
                            break
                    except:
                        continue
                if has_reducing_agent:
                    break
            
            # Return True if we have lactone -> lactol transformation
            # Reducing agent check is helpful but not required due to possible reagent omission
            return has_lactone_reactant and has_lactol_product
            
        except Exception:
            return False
