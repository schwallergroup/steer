"""Generated evaluation code for: Late stage nitro group removal"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNitroRemoval(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nitro group removal reactions.
    
    Detects nitro reduction or denitration reactions and scores based on how
    late in the synthesis they occur, with later reactions receiving higher scores.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        """
        Convert depth fraction to score (0-10).
        Later nitro removal (higher x) gets better score.
        """
        if x < 0:
            return 0  # No nitro removal reaction found
        
        if x >= self.stage_threshold:
            return 10  # Late-stage removal as desired
        else:
            # Scale score based on how close to late-stage threshold
            return 10 * (x / self.stage_threshold)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents nitro group removal.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Remove None molecules (parsing failures)
            reactants = [mol for mol in reactants if mol is not None]
            products = [mol for mol in products if mol is not None]
            
            if not reactants or not products:
                return False
            
            # Check for nitro group removal patterns
            return self._is_nitro_removal_reaction(reactants, products)
            
        except Exception:
            return False
    
    def _is_nitro_removal_reaction(self, reactants, products) -> bool:
        """
        Detect if reaction involves nitro group removal.
        """
        # Define nitro group patterns
        nitro_patterns = [
            "[c,C][N+](=O)[O-]",  # Nitro group (aromatic or aliphatic)
            "[c,C][N](=O)=O",     # Alternative nitro representation
        ]
        
        # Compile patterns
        nitro_mols = [Chem.MolFromSmarts(pattern) for pattern in nitro_patterns]
        nitro_mols = [mol for mol in nitro_mols if mol is not None]
        
        if not nitro_mols:
            return False
        
        # Count nitro groups in reactants and products
        reactant_nitro_count = sum(
            sum(mol.GetSubstructMatches(pattern) for pattern in nitro_mols)
            for mol in reactants
        )
        
        product_nitro_count = sum(
            sum(mol.GetSubstructMatches(pattern) for pattern in nitro_mols)
            for mol in products
        )
        
        # Check for reduction to amine (nitro -> amine)
        if reactant_nitro_count > product_nitro_count:
            return self._has_corresponding_amine_formation(reactants, products)
        
        # Check for denitration (complete removal)
        if reactant_nitro_count > product_nitro_count:
            return True
            
        return False
    
    def _has_corresponding_amine_formation(self, reactants, products) -> bool:
        """
        Check if nitro reduction leads to amine formation.
        """
        # Amine patterns (primary, secondary amines on aromatic/aliphatic carbons)
        amine_patterns = [
            "[c,C][NH2]",     # Primary amine
            "[c,C][NH][C,c]", # Secondary amine
        ]
        
        amine_mols = [Chem.MolFromSmarts(pattern) for pattern in amine_patterns]
        amine_mols = [mol for mol in amine_mols if mol is not None]
        
        if not amine_mols:
            return False
        
        # Count amines in reactants vs products
        reactant_amine_count = sum(
            sum(mol.GetSubstructMatches(pattern) for pattern in amine_mols)
            for mol in reactants
        )
        
        product_amine_count = sum(
            sum(mol.GetSubstructMatches(pattern) for pattern in amine_mols)
            for mol in products
        )
        
        # Expect increase in amine count when nitro is reduced
        return product_amine_count > reactant_amine_count
