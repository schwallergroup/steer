"""Generated evaluation code for: Late stage oxindole ring formation via reductive cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageOxindoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage oxindole ring formation via reductive cyclization.
    
    Detects when an oxindole ring (C1CC(=O)Nc2ccccc21) is formed through reductive cyclization,
    typically involving reduction of a nitro group followed by intramolecular amidation.
    Rewards routes where this formation occurs late in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.oxindole_pattern = Chem.MolFromSmarts(config["parameters"]["ring_smarts"])
        self.timing = config["parameters"]["timing"]
        self.formation_method = config["parameters"]["formation_method"]
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Oxindole formation doesn't occur
        else:
            # For late-stage formation, higher depth fraction is better
            # Convert to 0-10 scale where late formation scores higher
            return min(10, x * 10)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step forms an oxindole ring via reductive cyclization.
        """
        try:
            rxn = d["metadata"]["mapped_reaction_smiles"].split(">>")
            reactants_smiles = rxn[0]
            products_smiles = rxn[1]
            
            # Parse reactants and products
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check if oxindole ring is formed (present in products but not reactants)
            oxindole_in_products = any(mol.HasSubstructMatch(self.oxindole_pattern) for mol in products)
            oxindole_in_reactants = any(mol.HasSubstructMatch(self.oxindole_pattern) for mol in reactants)
            
            if not (oxindole_in_products and not oxindole_in_reactants):
                return False
            
            # Check for reductive cyclization pattern
            return self._is_reductive_cyclization(reactants, products)
            
        except Exception:
            return False
    
    def _is_reductive_cyclization(self, reactants, products) -> bool:
        """
        Check if the reaction involves reductive cyclization leading to oxindole formation.
        Look for nitro group reduction or similar reductive conditions with cyclization.
        """
        # Pattern for nitro group that could be reduced
        nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
        # Pattern for potential cyclization precursor (amide formation capability)
        amide_precursor_pattern = Chem.MolFromSmarts("C(=O)[OH,Cl,Br,I]")
        
        # Check if reactants contain nitro groups
        has_nitro_reactant = any(mol.HasSubstructMatch(nitro_pattern) for mol in reactants)
        
        # Check if reactants have amide-forming potential
        has_amide_precursor = any(mol.HasSubstructMatch(amide_precursor_pattern) for mol in reactants)
        
        # Check if products have fewer nitro groups (indicating reduction)
        nitro_count_reactants = sum(len(mol.GetSubstructMatches(nitro_pattern)) for mol in reactants)
        nitro_count_products = sum(len(mol.GetSubstructMatches(nitro_pattern)) for mol in products)
        
        # Look for common reducing agents or conditions
        reducing_agents = [
            Chem.MolFromSmarts("[Fe]"),  # Iron
            Chem.MolFromSmarts("[Zn]"),  # Zinc
            Chem.MolFromSmarts("[Sn]"),  # Tin
            Chem.MolFromSmarts("N"),     # NH4+ or similar
        ]
        
        has_reducing_agent = any(
            any(mol.HasSubstructMatch(agent) for mol in reactants)
            for agent in reducing_agents if agent
        )
        
        # Reductive cyclization criteria:
        # 1. Nitro group present in reactants and reduced in products, OR
        # 2. Presence of reducing conditions with cyclization capability
        reductive_conditions = (
            (has_nitro_reactant and nitro_count_products < nitro_count_reactants) or
            has_reducing_agent
        )
        
        # Must also have potential for cyclization (amide formation)
        return reductive_conditions and (has_amide_precursor or has_nitro_reactant)
