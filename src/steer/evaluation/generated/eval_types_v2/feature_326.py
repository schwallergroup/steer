"""Generated evaluation code for: Late stage Baeyer-Villiger oxidation for phenol formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BaeyerVilligerDepth(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Baeyer-Villiger oxidation reactions
    at a specified depth threshold. Baeyer-Villiger oxidation converts ketones to esters
    or lactones, often used for phenol formation via subsequent hydrolysis.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 7)
        self.stage = config.get("stage", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Baeyer-Villiger oxidation doesn't occur
        
        if self.stage == "late":
            # Reward later stage reactions (closer to target molecule)
            if x <= self.depth_threshold / 10.0:  # Convert to fraction
                return 10 - (x * 50)  # Higher score for earlier fraction (later stage)
            else:
                return 2  # Lower score if too early in synthesis
        else:
            # For early stage, reverse the scoring
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Detect Baeyer-Villiger oxidation by identifying ketone to ester/lactone conversion.
        Pattern: C(=O)C -> C(=O)OC (ester) or ring expansion of cyclic ketones.
        """
        rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
        if not rxn_smiles or ">>" not in rxn_smiles:
            return False
        
        try:
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            reactants = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactants or None in products:
                return False
            
            # Define patterns for Baeyer-Villiger oxidation
            ketone_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[C:3]")  # General ketone
            ester_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3][C:4]")  # Ester product
            cyclic_ketone_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[C:3]1~*~*~*~[C:4]1")  # Cyclic ketone (5-7 membered)
            lactone_pattern = Chem.MolFromSmarts("[C:1](=[O:2])[O:3]*1~*~*~*~[C:4]1")  # Lactone product
            
            # Check for ketone in reactants
            has_ketone = False
            has_cyclic_ketone = False
            for reactant in reactants:
                if reactant.HasSubstructMatch(ketone_pattern):
                    has_ketone = True
                if reactant.HasSubstructMatch(cyclic_ketone_pattern):
                    has_cyclic_ketone = True
            
            if not has_ketone:
                return False
            
            # Check for ester/lactone in products
            has_ester_product = False
            for product in products:
                if product.HasSubstructMatch(ester_pattern):
                    has_ester_product = True
                if has_cyclic_ketone and product.HasSubstructMatch(lactone_pattern):
                    has_ester_product = True
            
            # Additional check: ensure oxygen insertion (characteristic of Baeyer-Villiger)
            reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
            product_atoms = sum(mol.GetNumAtoms() for mol in products)
            
            # Baeyer-Villiger should show oxygen insertion without major structural changes
            return has_ester_product and (product_atoms - reactant_atoms) >= 0
            
        except Exception:
            return False
