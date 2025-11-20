"""Generated evaluation code for: Late stage alcohol dehydration"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageAlcoholDehydration(BaseScoring):
    """
    Evaluates synthesis routes for late-stage alcohol dehydration reactions.
    Specifically looks for dehydration of secondary alcohols to form alkenes,
    favoring reactions that occur in the final steps of the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "continuous")
        self.target_depth = config.get("target_depth", {}).get("value", 0.9)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to 0-10 score.
        Late-stage dehydration (high x values) gets higher scores.
        """
        if x < 0:
            return 0  # Dehydration doesn't occur
        else:
            # Favor late-stage dehydration - higher depth fraction gets better score
            return x * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents alcohol dehydration.
        Looks for secondary alcohol -> alkene transformation.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for secondary alcohol in reactants
            secondary_alcohol_pattern = Chem.MolFromSmarts("[CH1]([OH])")
            has_secondary_alcohol = any(
                mol.HasSubstructMatch(secondary_alcohol_pattern) for mol in reactants
            )
            
            if not has_secondary_alcohol:
                return False
            
            # Check for alkene formation in products (C=C double bond)
            alkene_pattern = Chem.MolFromSmarts("C=C")
            has_alkene = any(
                mol.HasSubstructMatch(alkene_pattern) for mol in products
            )
            
            if not has_alkene:
                return False
            
            # Additional check: ensure water is eliminated (dehydration signature)
            # Look for loss of OH and H to form water
            reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants)
            product_atoms = sum(mol.GetNumAtoms() for mol in products)
            
            # Should lose H2O (3 atoms) in dehydration
            atom_loss = reactant_atoms - product_atoms
            
            return has_secondary_alcohol and has_alkene and atom_loss >= 3
            
        except Exception:
            return False
