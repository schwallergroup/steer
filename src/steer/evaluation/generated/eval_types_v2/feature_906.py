"""Generated evaluation code for: Multiple Williamson ether formations for aryl linkages"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleWilliamsonEtherFormation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of multiple Williamson ether formations
    specifically targeting aryl ether linkages. Checks if the route contains the specified
    number of Williamson ether synthesis reactions forming aryl ether bonds.
    """
    
    def __init__(self, config):
        self.target_count = config.get("count", 4)
        self.substrate_type = config.get("substrate_type", "aryl")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        williamson_count = sum(1 for r in reactions if self.detect_williamson_ether_formation(r))
        
        # Condition is met if we have exactly the target count of Williamson ether formations
        condition = williamson_count == self.target_count
        return condition, len(reactions)
    
    def detect_williamson_ether_formation(self, rxn):
        """
        Detects Williamson ether synthesis reactions forming aryl ether bonds.
        Looks for C-O bond formation between aryl halide/tosylate and alkoxide/phenoxide.
        """
        try:
            reactants_smiles, products_smiles = rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for aryl halide pattern (Ar-X where X = Cl, Br, I, or tosylate)
            aryl_halide_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[Cl,Br,I]")
            tosylate_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[O:2]-S(=O)(=O)-c1ccc(C)cc1")
            
            # Check for alkoxide/phenoxide nucleophile patterns
            alkoxide_pattern = Chem.MolFromSmarts("[C:3]-[O-]")
            phenoxide_pattern = Chem.MolFromSmarts("[c:3]-[O-]")
            
            # Check for aryl ether product pattern
            aryl_ether_pattern = Chem.MolFromSmarts("[cH0,cH1:1]-[O:2]-[C,c:3]")
            
            has_aryl_halide = False
            has_nucleophile = False
            has_aryl_ether_product = False
            
            # Check reactants for aryl halide and nucleophile
            for reactant in reactants:
                if reactant.HasSubstructMatch(aryl_halide_pattern) or reactant.HasSubstructMatch(tosylate_pattern):
                    has_aryl_halide = True
                if reactant.HasSubstructMatch(alkoxide_pattern) or reactant.HasSubstructMatch(phenoxide_pattern):
                    has_nucleophile = True
            
            # Check products for aryl ether formation
            for product in products:
                if product.HasSubstructMatch(aryl_ether_pattern):
                    has_aryl_ether_product = True
            
            # Additional check: ensure we're forming a new C-O bond
            # Count aryl ethers in products vs reactants
            reactant_aryl_ethers = sum(len(r.GetSubstructMatches(aryl_ether_pattern)) for r in reactants)
            product_aryl_ethers = sum(len(p.GetSubstructMatches(aryl_ether_pattern)) for p in products)
            
            net_ether_formation = product_aryl_ethers > reactant_aryl_ethers
            
            return has_aryl_halide and has_nucleophile and has_aryl_ether_product and net_ether_formation
            
        except Exception:
            return False
    
    def route_scoring(self, x):
        """
        Scoring based on how close we get to the target count.
        Returns higher score for exact match, lower for deviation.
        """
        if x < 0:
            return 0  # Condition not met at all
        
        # x represents the fraction of reactions that are Williamson ether formations
        # Convert back to approximate count for scoring
        total_reactions = getattr(self, '_last_reaction_count', 10)  # fallback
        actual_count = int(x * total_reactions)
        
        if actual_count == self.target_count:
            return 10  # Perfect match
        else:
            # Penalize deviation from target
            deviation = abs(actual_count - self.target_count)
            return max(0, 10 - 2 * deviation)
