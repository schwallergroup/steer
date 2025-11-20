"""Generated evaluation code for: Early stage reductive amination coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyReductiveAmination(BaseScoring):
    """
    Evaluates whether a reductive amination reaction occurs early in the synthesis route.
    
    Reductive amination typically involves the formation of an imine intermediate followed by
    reduction to form a C-N bond. This class detects the characteristic pattern of amine +
    carbonyl -> secondary/tertiary amine.
    """
    
    def __init__(self, config: Dict):
        # Early stage means we want it to happen at low depth (close to 0)
        self.target_timing = config.get("timing", "early")
        self.penalty_factor = config.get("penalty_factor", 1.0)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.target_timing == "early":
            # Early stage preferred - lower depth fraction is better
            return max(0, 10 * (1 - x))  # Score decreases as depth increases
        else:
            # Late stage preferred - higher depth fraction is better
            return max(0, 10 * x)
    
    def hit_condition(self, d):
        """
        Detect reductive amination by checking for:
        1. Formation of C-N bond
        2. Presence of carbonyl in reactants
        3. Presence of amine in reactants
        4. Reduction conditions or hydride reagents
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        if ">>" not in rxn_smiles:
            return False
            
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
                
            # Check for reductive amination patterns
            return self._detect_reductive_amination(reactants, products)
            
        except Exception:
            return False
    
    def _detect_reductive_amination(self, reactants, products):
        """
        Detect reductive amination by checking for:
        - Carbonyl group (aldehyde/ketone) in reactants
        - Primary or secondary amine in reactants  
        - Formation of new C-N bond in products
        - Possible presence of reducing agents
        """
        # SMARTS patterns
        aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)[#6]")  # Aldehyde
        ketone_pattern = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")   # Ketone
        primary_amine_pattern = Chem.MolFromSmarts("[NX3;H2;!$(NC=O)]")  # Primary amine
        secondary_amine_pattern = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]") # Secondary amine
        hydride_pattern = Chem.MolFromSmarts("[#1-,BH4-,AlH4-]")  # Hydride sources
        
        # Check reactants for carbonyl and amine
        has_carbonyl = False
        has_amine = False
        has_hydride = False
        
        for reactant in reactants:
            if reactant.HasSubstructMatch(aldehyde_pattern) or reactant.HasSubstructMatch(ketone_pattern):
                has_carbonyl = True
            if reactant.HasSubstructMatch(primary_amine_pattern) or reactant.HasSubstructMatch(secondary_amine_pattern):
                has_amine = True
            if reactant.HasSubstructMatch(hydride_pattern):
                has_hydride = True
                
        # Also check for common reducing agents by name/formula
        reactant_smiles = [Chem.MolToSmiles(mol) for mol in reactants]
        reducing_agents = ['[BH4-]', '[AlH4-]', 'CC(C)C', 'O']  # NaBH4, LiAlH4, etc.
        has_reducer = any(agent in ' '.join(reactant_smiles) for agent in reducing_agents)
        
        # Check products for secondary/tertiary amine formation
        has_product_amine = False
        secondary_tertiary_amine = Chem.MolFromSmarts("[NX3;H1,H0;!$(NC=O)]")
        
        for product in products:
            if product.HasSubstructMatch(secondary_tertiary_amine):
                has_product_amine = True
                break
                
        # Reductive amination requires: carbonyl + amine -> amine product + reducing conditions
        return has_carbonyl and has_amine and has_product_amine and (has_hydride or has_reducer)
