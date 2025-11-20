"""Generated evaluation code for: Late stage Williamson ether synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateWilliamsonEther(BaseScoring):
    """
    Evaluates whether Williamson ether synthesis occurs at a late stage in the route.
    Williamson ether synthesis involves nucleophilic substitution of an alkoxide with 
    an alkyl halide or tosylate to form an ether bond (R-O-R').
    """
    
    def __init__(self, config: Dict):
        self.step_position_from_end = config.get("step_position_from_end", 1)
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        else:
            # Late-stage occurrence is better, perfect score if at target position
            target_fraction = (self.step_position_from_end - 1) / max(1, self.step_position_from_end)
            return max(0, 1 - abs(x - target_fraction))
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by identifying:
        1. Formation of new C-O bond
        2. Presence of alkoxide nucleophile and alkyl electrophile patterns
        3. Leaving group (halide, tosylate) departure
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            products = [Chem.MolFromSmiles(p) for p in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            # Check for ether formation pattern
            if not self._has_ether_formation(reactants, products):
                return False
            
            # Check for Williamson mechanism indicators
            return self._detect_williamson_mechanism(reactants, products)
            
        except:
            return False
    
    def _has_ether_formation(self, reactants, products):
        """Check if new ether bond (C-O-C) is formed"""
        # Count C-O-C patterns in reactants vs products
        ether_pattern = Chem.MolFromSmarts("[C]-[O]-[C]")
        
        reactant_ethers = sum(len(mol.GetSubstructMatches(ether_pattern)) 
                             for mol in reactants if mol)
        product_ethers = sum(len(mol.GetSubstructMatches(ether_pattern)) 
                            for mol in products if mol)
        
        return product_ethers > reactant_ethers
    
    def _detect_williamson_mechanism(self, reactants, products):
        """
        Detect characteristic patterns of Williamson ether synthesis:
        - Alkoxide nucleophile (often as metal salt)
        - Alkyl halide or sulfonate electrophile
        """
        # Patterns for alkoxide nucleophiles
        alkoxide_patterns = [
            "[C]-[O-]",  # Alkoxide anion
            "[C]-[O][Na,K,Cs,Li]",  # Metal alkoxide
            "[C]-[OH]"   # Alcohol (can be deprotonated in situ)
        ]
        
        # Patterns for electrophilic substrates
        electrophile_patterns = [
            "[C][Cl,Br,I]",  # Alkyl halides
            "[C]OS(=O)(=O)[C]",  # Alkyl tosylate
            "[C]OS(=O)(=O)[F,Cl,Br]",  # Other sulfonates
        ]
        
        has_alkoxide = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in alkoxide_patterns)
            for mol in reactants if mol
        )
        
        has_electrophile = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                for pattern in electrophile_patterns)
            for mol in reactants if mol
        )
        
        # Check for leaving group departure
        has_leaving_group_departure = self._detect_leaving_group_departure(reactants, products)
        
        return has_alkoxide and has_electrophile and has_leaving_group_departure
    
    def _detect_leaving_group_departure(self, reactants, products):
        """Check if typical leaving groups (halides, sulfonates) are eliminated"""
        leaving_groups = ["[Cl-]", "[Br-]", "[I-]", "OS(=O)(=O)[C]"]
        
        for lg_smarts in leaving_groups:
            lg_pattern = Chem.MolFromSmarts(lg_smarts)
            reactant_lg = sum(len(mol.GetSubstructMatches(lg_pattern)) 
                             for mol in reactants if mol)
            product_lg = sum(len(mol.GetSubstructMatches(lg_pattern)) 
                            for mol in products if mol)
            
            if product_lg > reactant_lg:  # Leaving group appears in products
                return True
        
        return False
