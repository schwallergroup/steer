"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateRingFormation(BaseScoring):
    """
    Evaluates whether a specific ring formation occurs late in the synthesis route.
    Detects when a target ring structure is formed by checking for its presence
    in products but absence in reactants.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "early" or "late"
        self.direction = config["parameters"]["direction"]  # "formation" or "break"
        
        # Convert SMARTS pattern to mol object for substructure matching
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        if self.ring_pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {self.ring_smarts}")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation gets higher score (closer to 1)
        else:  # early timing
            return x  # Earlier formation gets higher score (x closer to 0 gets higher score)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves formation of the target ring.
        Ring formation detected by: ring present in product but absent in reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            product_smiles = rxn_parts[0].strip()
            reactants_smiles = rxn_parts[1].strip()
            
            # Parse product
            product_mol = Chem.MolFromSmiles(product_smiles)
            if product_mol is None:
                return False
                
            # Parse reactants
            reactant_mols = []
            for r_smiles in reactants_smiles.split("."):
                r_mol = Chem.MolFromSmiles(r_smiles.strip())
                if r_mol is not None:
                    reactant_mols.append(r_mol)
            
            if not reactant_mols:
                return False
            
            # Check if ring is present in product
            ring_in_product = product_mol.HasSubstructMatch(self.ring_pattern)
            
            if self.direction == "formation":
                if not ring_in_product:
                    return False  # Ring must be in product for formation
                    
                # Check if ring is absent in all reactants (indicating formation)
                ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
                return not ring_in_reactants  # Formation: ring in product but not in reactants
                
            else:  # direction == "break"
                if ring_in_product:
                    return False  # Ring must be absent in product for breaking
                    
                # Check if ring is present in any reactant (indicating breaking)
                ring_in_reactants = any(mol.HasSubstructMatch(self.ring_pattern) for mol in reactant_mols)
                return ring_in_reactants  # Breaking: ring in reactants but not in product
                
        except Exception:
            return False
