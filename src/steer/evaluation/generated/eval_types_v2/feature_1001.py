"""Generated evaluation code for: Early cyclopropane ring formation via Corey-Chaykovsky"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyCyclopropaneCoreyChaykovskyFormation(BaseScoring):
    """
    Evaluates if cyclopropane ring formation occurs early in the synthesis route
    using the Corey-Chaykovsky reaction method.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CC1"
        self.timing = config["parameters"]["timing"]  # "early"
        self.depth_threshold = config["parameters"]["depth_threshold"]  # 2
        self.reaction_method = config["parameters"]["reaction_method"]  # "corey_chaykovsky"
        
        # Compile the cyclopropane pattern
        self.cyclopropane_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
        # Corey-Chaykovsky reaction patterns - sulfonium ylide cyclopropanation
        self.corey_chaykovsky_patterns = [
            # Sulfonium ylide pattern
            Chem.MolFromSmarts("[S+]([C-])[C,c]"),
            # Alternative dimethylsulfonium methylide pattern
            Chem.MolFromSmarts("[S+]([CH2-])(C)C"),
            # Generic ylide carbon pattern
            Chem.MolFromSmarts("[C-][S+]")
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Condition not met
        
        # For early timing, lower depth values get higher scores
        if self.timing == "early":
            if x <= self.depth_threshold / 10.0:  # x is depth fraction
                return 10  # Perfect score for very early formation
            else:
                # Decreasing score as depth increases
                return max(0, 10 * (1 - x))
        else:
            # Default scoring
            return 10 * (1 - x)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves cyclopropane formation via Corey-Chaykovsky method
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for r_smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smi.strip())
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for p_smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smi.strip())
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if cyclopropane is formed (present in products but not in reactants)
            has_cyclopropane_in_products = any(
                mol.HasSubstructMatch(self.cyclopropane_pattern) for mol in product_mols
            )
            
            has_cyclopropane_in_reactants = any(
                mol.HasSubstructMatch(self.cyclopropane_pattern) for mol in reactant_mols
            )
            
            # Cyclopropane must be formed in this step
            if not (has_cyclopropane_in_products and not has_cyclopropane_in_reactants):
                return False
            
            # Check if Corey-Chaykovsky reagents are present
            has_corey_chaykovsky_reagent = False
            for reactant in reactant_mols:
                for pattern in self.corey_chaykovsky_patterns:
                    if reactant.HasSubstructMatch(pattern):
                        has_corey_chaykovsky_reagent = True
                        break
                if has_corey_chaykovsky_reagent:
                    break
            
            return has_corey_chaykovsky_reagent
            
        except Exception:
            return False
