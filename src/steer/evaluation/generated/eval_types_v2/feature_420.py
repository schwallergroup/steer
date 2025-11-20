"""Generated evaluation code for: Late pyrrolidine ring formation via cyclization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class PyrrolidineRingFormationTiming(BaseScoring):
    """
    Evaluates timing of pyrrolidine ring formation in synthesis routes.
    Rewards late-stage cyclization reactions that form pyrrolidine rings.
    """
    
    def __init__(self, config: Dict):
        self.ring_smarts = config["parameters"]["ring_smarts"]  # "C1CCCN1"
        self.timing = config["parameters"]["timing"]  # "late"
        self.direction = config["parameters"]["direction"]  # "formation"
        self.pyrrolidine_pattern = Chem.MolFromSmarts(self.ring_smarts)
    
    def route_scoring(self, x) -> float:
        """
        Score based on depth where pyrrolidine formation occurs.
        For late timing: later formation (higher depth fraction) gets better score.
        """
        if x < 0:
            return 0  # Ring formation doesn't happen
        
        if self.timing == "late":
            return 1 - x  # Later formation is better (closer to 1.0 depth)
        elif self.timing == "early":
            return x  # Earlier formation is better (closer to 0.0 depth)
        else:
            return 0.5  # Neutral scoring if timing not specified
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction step involves pyrrolidine ring formation.
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
            reactants = []
            for r_smiles in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(r_smiles)
                if mol is not None:
                    reactants.append(mol)
            
            products = []
            for p_smiles in products_smiles.split("."):
                mol = Chem.MolFromSmiles(p_smiles)
                if mol is not None:
                    products.append(mol)
            
            if not reactants or not products:
                return False
            
            # Count pyrrolidine rings in reactants vs products
            reactant_rings = sum(len(mol.GetSubstructMatches(self.pyrrolidine_pattern)) 
                               for mol in reactants)
            product_rings = sum(len(mol.GetSubstructMatches(self.pyrrolidine_pattern)) 
                              for mol in products)
            
            if self.direction == "formation":
                # Ring formation: more rings in products than reactants
                return product_rings > reactant_rings
            elif self.direction == "breaking":
                # Ring breaking: fewer rings in products than reactants
                return product_rings < reactant_rings
            else:
                return False
                
        except Exception:
            return False
