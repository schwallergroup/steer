"""Generated evaluation code for: Ketone to amine conversion via reductive amination"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ReductiveAmination(BaseScoring):
    """
    Evaluates synthesis routes for the presence of reductive amination reactions.
    Checks for ketone to amine conversion via reductive amination at any depth in the route.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "bool")
        self.target_depth = config.get("target_depth", {}).get("value", -1)
    
    def route_scoring(self, x) -> float:
        if self.condition_type == "bool":
            if self.target_depth == -1:  # Positive if condition is met
                return 1 if x >= 0 else 0
        else:
            if x < 0:
                return 0  # Reaction not found
            return max(0, 1 - abs(x - self.target_depth) / 10)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction node represents a reductive amination.
        Looks for ketone disappearing and primary amine appearing.
        """
        try:
            mapped_rxn = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not mapped_rxn or ">>" not in mapped_rxn:
                return False
            
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Define SMARTS patterns
            ketone_pattern = Chem.MolFromSmarts("[C;X3]=[O;X1]")  # Ketone carbonyl
            primary_amine_pattern = Chem.MolFromSmarts("[C][NH2]")  # Primary amine
            
            # Check for ketone in reactants
            has_ketone_reactant = any(mol.HasSubstructMatch(ketone_pattern) for mol in reactants)
            
            # Check for primary amine in products
            has_amine_product = any(mol.HasSubstructMatch(primary_amine_pattern) for mol in products)
            
            # Check for reducing agents (common reductive amination reagents)
            reducing_agents = [
                "[B-]([H])([H])([H])[H]",  # Borohydride
                "[Al]([H])([H])[H]",       # Aluminum hydride
                "[H][H]"                   # H2 gas
            ]
            
            has_reducing_agent = False
            for agent_smarts in reducing_agents:
                agent_pattern = Chem.MolFromSmarts(agent_smarts)
                if agent_pattern and any(mol.HasSubstructMatch(agent_pattern) for mol in reactants):
                    has_reducing_agent = True
                    break
            
            # Alternative check: look for nitrogen source (ammonia, ammonium salts)
            nitrogen_sources = [
                "[NH3]",           # Ammonia
                "[NH4+]",          # Ammonium ion
                "[N]([H])([H])[H]" # Ammonia alternative
            ]
            
            has_nitrogen_source = False
            for n_smarts in nitrogen_sources:
                n_pattern = Chem.MolFromSmarts(n_smarts)
                if n_pattern and any(mol.HasSubstructMatch(n_pattern) for mol in reactants):
                    has_nitrogen_source = True
                    break
            
            # Reductive amination criteria:
            # 1. Ketone in reactants
            # 2. Primary amine in products  
            # 3. Either reducing agent OR nitrogen source present
            return (has_ketone_reactant and 
                   has_amine_product and 
                   (has_reducing_agent or has_nitrogen_source))
            
        except Exception:
            return False
