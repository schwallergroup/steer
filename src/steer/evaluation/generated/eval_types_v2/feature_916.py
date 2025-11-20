"""Generated evaluation code for: Late thiazole ring formation via Hantzsch synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateThiazoleHantzsch(BaseScoring):
    """
    Evaluates synthesis routes for late-stage thiazole ring formation via Hantzsch synthesis.
    
    The Hantzsch thiazole synthesis typically involves the reaction of a thioamide 
    with an α-haloketone or α-haloaldehyde to form the thiazole ring. This class
    checks for thiazole ring formation and rewards it when it occurs later in the synthesis.
    """
    
    def __init__(self, config: Dict):
        self.thiazole_smarts = config["parameters"]["ring_smarts"]  # "c1scnc1"
        self.thiazole_pattern = Chem.MolFromSmarts(self.thiazole_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Score based on depth fraction where thiazole formation occurs.
        Late-stage formation (higher x) gets better score.
        """
        if x < 0:
            return 0  # No thiazole formation found
        else:
            return x * 10  # Later formation gets higher score (0-10 scale)
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction involves thiazole ring formation consistent with Hantzsch synthesis.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check if product contains thiazole ring
            if not product_mol.HasSubstructMatch(self.thiazole_pattern):
                return False
            
            # Check that reactants don't already have thiazole (ring formation, not just modification)
            reactant_mols = []
            for r_smiles in reactant_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles)
                if r_mol:
                    reactant_mols.append(r_mol)
            
            # Ensure thiazole ring is being formed (not present in reactants)
            thiazole_in_reactants = any(mol.HasSubstructMatch(self.thiazole_pattern) for mol in reactant_mols)
            if thiazole_in_reactants:
                return False
            
            # Check for Hantzsch synthesis pattern: thioamide + α-haloketone/aldehyde
            thioamide_pattern = Chem.MolFromSmarts("[#6]-[#6](=[#16])-[#7]")  # R-C(=S)-N
            halo_carbonyl_pattern = Chem.MolFromSmarts("[#6]-[#6](=[#8])-[#6]-[#9,#17,#35,#53]")  # R-CO-CH-X
            
            has_thioamide = any(mol.HasSubstructMatch(thioamide_pattern) for mol in reactant_mols)
            has_halo_carbonyl = any(mol.HasSubstructMatch(halo_carbonyl_pattern) for mol in reactant_mols)
            
            return has_thioamide and has_halo_carbonyl
            
        except Exception:
            return False
