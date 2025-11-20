"""Generated evaluation code for: Late stage cyclopropanation of vinyl amide"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageVinylAmideCyclopropanation(BaseScoring):
    """
    Evaluates routes for late-stage cyclopropanation reactions on vinyl amide substrates.
    Rewards routes where cyclopropanation occurs later in the synthesis on complex substrates.
    """
    
    def __init__(self, config: Dict):
        # Vinyl amide pattern for substrate matching
        self.vinyl_amide_pattern = Chem.MolFromSmarts("C=C-C(=O)N")
        # Cyclopropane pattern to detect in products
        self.cyclopropane_pattern = Chem.MolFromSmarts("C1CC1")
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where cyclopropanation occurs.
        Later stage (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # No cyclopropanation found
        else:
            # Late-stage cyclopropanation is better (closer to 1.0 depth fraction)
            return 10 * x
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step represents cyclopropanation of vinyl amide.
        """
        try:
            # Parse reaction SMILES
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
                
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Check if any reactant has vinyl amide pattern
            has_vinyl_amide_reactant = any(
                mol.HasSubstructMatch(self.vinyl_amide_pattern) 
                for mol in reactant_mols
            )
            
            if not has_vinyl_amide_reactant:
                return False
            
            # Check if any product has cyclopropane pattern
            has_cyclopropane_product = any(
                mol.HasSubstructMatch(self.cyclopropane_pattern)
                for mol in product_mols
            )
            
            # Check if cyclopropane was formed (not present in reactants)
            cyclopropane_in_reactants = any(
                mol.HasSubstructMatch(self.cyclopropane_pattern)
                for mol in reactant_mols
            )
            
            # True if we have vinyl amide substrate, form cyclopropane, and it wasn't already there
            return has_vinyl_amide_reactant and has_cyclopropane_product and not cyclopropane_in_reactants
            
        except Exception:
            return False
