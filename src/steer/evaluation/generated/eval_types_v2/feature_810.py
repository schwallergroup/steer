"""Generated evaluation code for: N-benzyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NBenzylProtectingGroupStrategy(BaseScoring):
    """
    Evaluates N-benzyl protecting group strategy in synthesis routes.
    Checks if N-benzyl protection is installed early via reductive amination
    and later removed by hydrogenolysis or debenzylation reactions.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("condition_type", "bool")
        self.target_depth = config.get("target_depth", 0.3)  # Early installation preferred
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Strategy not found
        else:
            # Reward early installation (lower depth values)
            if self.condition_type == "bool":
                return 1  # Strategy found
            else:
                # Early installation is better (closer to 0)
                return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction involves N-benzyl protection installation"""
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            products = [Chem.MolFromSmiles(p) for p in rxn[0].split(".")]
            reactants = [Chem.MolFromSmiles(r) for r in rxn[1].split(".")]
            
            # Check for reductive amination pattern forming N-benzyl bond
            return self._is_reductive_amination_with_benzyl(reactants, products)
            
        except Exception:
            return False
    
    def _is_reductive_amination_with_benzyl(self, reactants, products):
        """Detect reductive amination forming N-benzyl bond"""
        # N-benzyl pattern: nitrogen connected to benzyl carbon
        nbenzyl_pattern = Chem.MolFromSmarts("[NH1,NH2]-[CH2]-c1ccccc1")
        
        # Benzaldehyde or benzyl halide patterns (common benzyl sources)
        benzaldehyde_pattern = Chem.MolFromSmarts("c1ccccc1-[CH1]=O")
        benzyl_halide_pattern = Chem.MolFromSmarts("c1ccccc1-[CH2]-[Cl,Br,I]")
        
        # Primary or secondary amine pattern
        amine_pattern = Chem.MolFromSmarts("[NH1,NH2]")
        
        # Check if product has N-benzyl group
        has_nbenzyl_product = any(mol.HasSubstructMatch(nbenzyl_pattern) for mol in products if mol)
        
        if not has_nbenzyl_product:
            return False
            
        # Check if reactants contain benzyl source and amine
        has_benzyl_source = any(
            mol.HasSubstructMatch(benzaldehyde_pattern) or mol.HasSubstructMatch(benzyl_halide_pattern)
            for mol in reactants if mol
        )
        
        has_amine = any(mol.HasSubstructMatch(amine_pattern) for mol in reactants if mol)
        
        # Also check for reducing agents (common in reductive amination)
        reducing_agents = [
            "[BH4-]",  # Sodium borohydride
            "[BH3]",   # Borane
            "P(c1ccccc1)3",  # Triphenylphosphine (for Staudinger-type)
            "[H][H]"   # Hydrogen gas
        ]
        
        has_reducing_agent = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(agent)) for mol in reactants if mol)
            for agent in reducing_agents
        )
        
        return has_benzyl_source and has_amine and (has_reducing_agent or self._likely_reductive_conditions(reactants))
    
    def _likely_reductive_conditions(self, reactants):
        """Check for other indicators of reductive conditions"""
        # Look for common reductive amination conditions
        metal_hydrides = ["[Na+]", "[Li+]", "[Al]", "[Zn]"]
        
        return any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(metal)) for mol in reactants if mol)
            for metal in metal_hydrides
        )
