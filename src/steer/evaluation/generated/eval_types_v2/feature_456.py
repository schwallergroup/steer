"""Generated evaluation code for: Late imidazole ring formation via condensation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateImidazoleCondensation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage imidazole ring formation via condensation.
    Detects when imidazole rings are formed through condensation reactions and rewards
    routes where this occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score.
        For late-stage preference: higher depth fraction = better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        else:
            # Late-stage formation is better, so return depth fraction scaled to 0-10
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents imidazole ring formation via condensation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
        
        # Split reaction into reactants and products
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        products_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        try:
            # Parse molecules
            product_mol = Chem.MolFromSmiles(products_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check if product contains imidazole ring
            if not product_mol.HasSubstructMatch(self.ring_pattern):
                return False
            
            # Check if any reactant already has the complete imidazole ring
            for reactant in reactant_mols:
                if reactant.HasSubstructMatch(self.ring_pattern):
                    return False  # Ring already exists, not a formation
            
            # Check for condensation pattern (formation of new C-N bonds with water elimination)
            if self._is_condensation_reaction(product_mol, reactant_mols):
                return True
            
        except Exception:
            return False
        
        return False
    
    def _is_condensation_reaction(self, product_mol, reactant_mols) -> bool:
        """
        Check if the reaction represents a condensation by looking for:
        1. Formation of new bonds in the imidazole ring
        2. Typical condensation patterns (amino + carbonyl/carboxyl groups)
        """
        # Simple heuristic: check for presence of amino and carbonyl/carboxyl groups in reactants
        amino_pattern = Chem.MolFromSmarts("[NH2,NH1]")
        carbonyl_pattern = Chem.MolFromSmarts("[C](=O)")
        carboxyl_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        
        has_amino = False
        has_carbonyl_or_carboxyl = False
        
        for reactant in reactant_mols:
            if reactant.HasSubstructMatch(amino_pattern):
                has_amino = True
            if reactant.HasSubstructMatch(carbonyl_pattern) or reactant.HasSubstructMatch(carboxyl_pattern):
                has_carbonyl_or_carboxyl = True
        
        # For imidazole formation, we typically need both amino and carbonyl/carboxyl groups
        return has_amino and has_carbonyl_or_carboxyl
