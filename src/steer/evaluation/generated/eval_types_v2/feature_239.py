"""Generated evaluation code for: Late stage C-N bond formation via alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAmineAlkylation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage C-N bond formation via amine alkylation.
    Detects nucleophilic substitution reactions where an amine attacks an alkyl halide
    or similar electrophile, particularly targeting reactions that occur in the final
    stages of synthesis (within depth threshold of 2).
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 2)
        self.timing = config.get("timing", "late")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        
        if self.timing == "late":
            # Reward reactions occurring at shallow depths (late in synthesis)
            if x <= self.depth_threshold / 10.0:  # Convert to depth fraction
                return 10  # Perfect score for very late stage
            else:
                return max(0, 10 - (x * 20))  # Penalize deeper reactions
        else:
            # General case - any occurrence gets points
            return 10 if x >= 0 else 0
    
    def hit_condition(self, d) -> bool:
        """
        Detects amine alkylation reactions by checking for:
        1. Formation of new C-N bond
        2. Presence of amine nucleophile in reactants
        3. Presence of alkyl electrophile (halide, tosylate, etc.) in reactants
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for amine nucleophile in reactants
            amine_patterns = [
                "[NH2]",  # Primary amine
                "[NH1]([#6])",  # Secondary amine
                "[NH0]([#6])([#6])",  # Tertiary amine
                "n",  # Aromatic nitrogen (can be nucleophilic)
            ]
            
            has_amine = False
            for reactant in reactants:
                for pattern in amine_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_amine = True
                        break
                if has_amine:
                    break
            
            if not has_amine:
                return False
            
            # Check for alkyl electrophile patterns
            electrophile_patterns = [
                "[#6][Cl]",  # Alkyl chloride
                "[#6][Br]",  # Alkyl bromide  
                "[#6][I]",   # Alkyl iodide
                "[#6]OS(=O)(=O)[#6]",  # Tosylate/mesylate
                "[#6]OS(=O)(=O)c1ccc(C)cc1",  # Tosylate specific
            ]
            
            has_electrophile = False
            for reactant in reactants:
                for pattern in electrophile_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_electrophile = True
                        break
                if has_electrophile:
                    break
            
            if not has_electrophile:
                return False
            
            # Verify C-N bond formation by checking if product has more C-N bonds
            # than the sum of reactants
            product_cn_bonds = len(product.GetSubstructMatches(Chem.MolFromSmarts("[#6]-[#7]")))
            reactant_cn_bonds = sum(len(r.GetSubstructMatches(Chem.MolFromSmarts("[#6]-[#7]"))) 
                                   for r in reactants)
            
            return product_cn_bonds > reactant_cn_bonds
            
        except Exception:
            return False
