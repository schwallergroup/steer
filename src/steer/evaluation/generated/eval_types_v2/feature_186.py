"""Generated evaluation code for: Late thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class HantzschThiazoleFormation(BaseScoring):
    """
    Evaluates routes based on when thiazole ring formation occurs via Hantzsch synthesis.
    Penalizes routes where thiazole formation happens later than the target stage.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.reaction_type = config["parameters"]["reaction_type"]  # "hantzsch_thiazole"
        self.target_stage = config["parameters"]["stage"]  # "early"
        
        # Convert stage preference to depth scoring
        self.early_preferred = self.target_stage == "early"
        
        # Compile SMARTS pattern for thiazole detection
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Thiazole formation doesn't occur
        
        if self.early_preferred:
            # Early formation preferred - penalize late formation
            return 1 - x  # x=0 (early) gives score 1, x=1 (late) gives score 0
        else:
            # Late formation preferred - penalize early formation  
            return x  # x=0 (early) gives score 0, x=1 (late) gives score 1
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents Hantzsch thiazole formation.
        Detects thiazole ring formation by checking if thiazole appears in products
        but not in reactants.
        """
        try:
            # Get mapped reaction SMILES
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactants = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol:
                    reactants.append(mol)
            
            products = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Check if thiazole ring is formed (present in products but not reactants)
            thiazole_in_reactants = any(mol.HasSubstructMatch(self.thiazole_pattern) for mol in reactants)
            thiazole_in_products = any(mol.HasSubstructMatch(self.thiazole_pattern) for mol in products)
            
            # Thiazole formation occurs if it's in products but not in reactants
            is_thiazole_formation = thiazole_in_products and not thiazole_in_reactants
            
            # Additional check for Hantzsch-like pattern (optional - can be enhanced)
            # Look for typical Hantzsch reactants: carbonyl + thiocarbonyl + amine
            is_hantzsch_like = self._detect_hantzsch_pattern(reactants)
            
            return is_thiazole_formation and is_hantzsch_like
            
        except Exception:
            return False
    
    def _detect_hantzsch_pattern(self, reactants) -> bool:
        """
        Detect if reactants contain typical Hantzsch synthesis components:
        - Alpha-halocarbonyl or similar electrophile
        - Thiocarbonyl/thioamide compound
        - Amine or similar nucleophile
        """
        # Simplified detection - look for key functional groups
        has_carbonyl = False
        has_sulfur_hetero = False
        has_nitrogen = False
        
        for mol in reactants:
            # Check for carbonyl groups
            carbonyl_pattern = Chem.MolFromSmarts("[#6]=[#8]")
            if mol.HasSubstructMatch(carbonyl_pattern):
                has_carbonyl = True
            
            # Check for sulfur-containing groups (thioamide, etc.)
            sulfur_patterns = [
                Chem.MolFromSmarts("[#16]"),  # Any sulfur
                Chem.MolFromSmarts("[#6]=[#16]"),  # Thiocarbonyl
            ]
            if any(mol.HasSubstructMatch(pattern) for pattern in sulfur_patterns):
                has_sulfur_hetero = True
            
            # Check for nitrogen nucleophiles
            nitrogen_patterns = [
                Chem.MolFromSmarts("[#7]"),  # Any nitrogen
                Chem.MolFromSmarts("[#7H2]"),  # Primary amine
                Chem.MolFromSmarts("[#7H1]"),  # Secondary amine
            ]
            if any(mol.HasSubstructMatch(pattern) for pattern in nitrogen_patterns):
                has_nitrogen = True
        
        return has_carbonyl and has_sulfur_hetero and has_nitrogen
