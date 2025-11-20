"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates whether a nucleophilic aromatic substitution (SNAr) reaction 
    occurs in the late stages of synthesis (within step_threshold steps from target).
    Returns higher scores for later occurrence of SNAr reactions.
    """
    
    def __init__(self, config: Dict):
        self.step_threshold = config["parameters"].get("step_threshold", 3)
        
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Late stage reactions (x close to 1) get higher scores.
        """
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        
        # Calculate steps from target based on depth fraction
        # Assume typical route depth of 10 steps for conversion
        estimated_steps_from_target = (1 - x) * 10
        
        if estimated_steps_from_target <= self.step_threshold:
            # Late stage - higher score for closer to target
            return 8 + 2 * (1 - estimated_steps_from_target / self.step_threshold)
        else:
            # Early stage - lower score
            return max(0, 5 - estimated_steps_from_target / 2)
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction is a nucleophilic aromatic substitution.
        Identifies SNAr by detecting aromatic carbon with leaving group being replaced.
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            
            if not all(products) or not all(reactants):
                return False
            
            # Look for nucleophilic aromatic substitution pattern
            return self._detect_snar_pattern(reactants, products)
            
        except Exception:
            return False
    
    def _detect_snar_pattern(self, reactants, products) -> bool:
        """
        Detect SNAr by identifying:
        1. Aromatic carbon that changes substitution
        2. Common leaving groups (OMe, F, Cl, Br, I, NO2)
        3. Common nucleophiles (NH2, NR2, OH, OR, etc.)
        """
        # Common SNAr patterns - aromatic carbons with leaving groups
        leaving_group_patterns = [
            "[cH0:1]([F,Cl,Br,I])",  # Halogen on aromatic carbon
            "[cH0:1](OC)",           # Methoxy on aromatic carbon  
            "[cH0:1](O)",            # Other alkoxy on aromatic carbon
            "[cH0:1]([N+](=O)[O-])", # Nitro group on aromatic carbon
        ]
        
        # Common nucleophile patterns
        nucleophile_patterns = [
            "[NH2]",           # Primary amine
            "[NH1]",           # Secondary amine
            "[NH0]",           # Tertiary amine
            "[OH]",            # Hydroxyl
            "[O-]",            # Alkoxide
            "N1CCNCC1",        # Piperazine (from rationale)
            "N1CCCCC1",        # Piperidine
        ]
        
        # Check if any reactant has leaving group pattern
        has_leaving_group = False
        for reactant in reactants:
            for lg_pattern in leaving_group_patterns:
                pattern_mol = Chem.MolFromSmarts(lg_pattern)
                if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                    has_leaving_group = True
                    break
            if has_leaving_group:
                break
        
        # Check if any reactant is a nucleophile
        has_nucleophile = False
        for reactant in reactants:
            for nu_pattern in nucleophile_patterns:
                pattern_mol = Chem.MolFromSmarts(nu_pattern)
                if pattern_mol and reactant.HasSubstructMatch(pattern_mol):
                    has_nucleophile = True
                    break
            if has_nucleophile:
                break
        
        # Check if product has aromatic C-N or C-O bond (common SNAr products)
        has_snar_product = False
        for product in products:
            snar_product_patterns = [
                "[cH0]-[NH2,NH1,NH0]",  # Aromatic C-N bond
                "[cH0]-[OH,O]",         # Aromatic C-O bond
            ]
            for prod_pattern in snar_product_patterns:
                pattern_mol = Chem.MolFromSmarts(prod_pattern)
                if pattern_mol and product.HasSubstructMatch(pattern_mol):
                    has_snar_product = True
                    break
            if has_snar_product:
                break
        
        return has_leaving_group and has_nucleophile and has_snar_product
