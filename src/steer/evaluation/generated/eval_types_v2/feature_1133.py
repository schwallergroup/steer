"""Generated evaluation code for: Late pyrazole ring formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LatePyrazoleFormation(BaseScoring):
    """
    Evaluates synthesis routes for late-stage pyrazole ring formation.
    Rewards routes where pyrazole rings are formed in the later stages of synthesis,
    typically via cyclization reactions like Paal-Knorr condensation.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]
        self.timing = config["parameters"]["timing"]  # "late" 
        self.direction = config["parameters"]["direction"]  # "formation"
        self.ring_pattern = Chem.MolFromSmarts(self.ring_smarts)
        
    def route_scoring(self, x) -> float:
        """
        Scoring function that rewards late-stage pyrazole formation.
        x is the depth fraction where pyrazole formation occurs.
        """
        if x < 0:
            return 0  # No pyrazole formation found
        
        if self.timing == "late":
            # Late formation is better - reward smaller depth fractions (closer to target)
            return 10 * (1 - x)  # Score 10 for depth 0, decreasing to 0 for depth 1
        else:
            # Early formation preferred
            return 10 * x
            
    def hit_condition(self, d) -> bool:
        """
        Checks if a reaction involves pyrazole ring formation.
        Returns True if pyrazole ring is formed in this reaction step.
        """
        if "mapped_reaction_smiles" not in d.get("metadata", {}):
            return False
            
        rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    reactant_mols.append(mol)
                    
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is not None:
                    product_mols.append(mol)
                    
            if not reactant_mols or not product_mols:
                return False
                
            # Count pyrazole rings in reactants and products
            reactant_pyrazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern)) 
                for mol in reactant_mols
            )
            
            product_pyrazole_count = sum(
                len(mol.GetSubstructMatches(self.ring_pattern))
                for mol in product_mols
            )
            
            # Check for ring formation (more pyrazole rings in products than reactants)
            if self.direction == "formation":
                return product_pyrazole_count > reactant_pyrazole_count
            else:  # "breaking"
                return reactant_pyrazole_count > product_pyrazole_count
                
        except Exception:
            return False
