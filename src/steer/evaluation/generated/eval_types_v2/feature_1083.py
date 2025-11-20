"""Generated evaluation code for: Late stage thiazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageThiazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation via Hantzsch synthesis.
    Rewards routes where thiazole rings (c1scnc1) are formed in later steps rather than 
    using pre-formed thiazole starting materials.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]
        self.method = config["parameters"]["method"]
        self.thiazole_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Converts depth fraction to score. For late-stage timing, later formation is better.
        Returns 0-10 scale where 10 is optimal timing.
        """
        if x < 0:
            return 0  # No thiazole formation found
        
        if self.timing == "late":
            # Later formation is better - reward higher depth fractions
            return 10 * x
        else:
            # Early formation preferred
            return 10 * (1 - x)
    
    def hit_condition(self, d) -> bool:
        """
        Checks if this reaction step involves thiazole ring formation.
        Returns True if thiazole ring is formed in this step.
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        # Parse reactants and products
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Count thiazole rings in reactants
            reactant_thiazoles = 0
            if "." in reactants_smiles:
                for r_smiles in reactants_smiles.split("."):
                    r_mol = Chem.MolFromSmiles(r_smiles)
                    if r_mol and r_mol.HasSubstructMatch(self.thiazole_pattern):
                        reactant_thiazoles += len(r_mol.GetSubstructMatches(self.thiazole_pattern))
            else:
                r_mol = Chem.MolFromSmiles(reactants_smiles)
                if r_mol and r_mol.HasSubstructMatch(self.thiazole_pattern):
                    reactant_thiazoles = len(r_mol.GetSubstructMatches(self.thiazole_pattern))
            
            # Count thiazole rings in products
            product_thiazoles = 0
            if "." in products_smiles:
                for p_smiles in products_smiles.split("."):
                    p_mol = Chem.MolFromSmiles(p_smiles)
                    if p_mol and p_mol.HasSubstructMatch(self.thiazole_pattern):
                        product_thiazoles += len(p_mol.GetSubstructMatches(self.thiazole_pattern))
            else:
                p_mol = Chem.MolFromSmiles(products_smiles)
                if p_mol and p_mol.HasSubstructMatch(self.thiazole_pattern):
                    product_thiazoles = len(p_mol.GetSubstructMatches(self.thiazole_pattern))
            
            # Check if net thiazole rings were formed (more in products than reactants)
            return product_thiazoles > reactant_thiazoles
            
        except Exception:
            return False
    
    def _is_hantzsch_like_reaction(self, reactants_smiles: str) -> bool:
        """
        Helper method to check if the reaction pattern resembles Hantzsch synthesis.
        Hantzsch synthesis typically involves α-haloketones and thiourea/thioamides.
        """
        try:
            # Common Hantzsch reactant patterns
            haloketone_pattern = Chem.MolFromSmarts("[#6][C](=O)[CH2][Cl,Br,I]")  # α-haloketone
            thiourea_pattern = Chem.MolFromSmarts("[NH2]C(=S)[NH2]")  # thiourea
            thioamide_pattern = Chem.MolFromSmarts("[#6]C(=S)[NH2]")  # thioamide
            
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol:
                    reactant_mols.append(mol)
            
            has_haloketone = any(mol.HasSubstructMatch(haloketone_pattern) for mol in reactant_mols)
            has_thiourea = any(mol.HasSubstructMatch(thiourea_pattern) or 
                             mol.HasSubstructMatch(thioamide_pattern) for mol in reactant_mols)
            
            return has_haloketone and has_thiourea
            
        except Exception:
            return False
