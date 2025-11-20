"""Generated evaluation code for: Late stage N-alkylation coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNAlkylation(BaseScoring):
    """
    Evaluates whether N-alkylation occurs at late stage in the synthesis route.
    Rewards routes where N-alkylation happens in the final steps.
    """
    
    def __init__(self, config: Dict):
        self.step_position = config.get("parameters", {}).get("step_position", "final")
        self.timing = config.get("parameters", {}).get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-alkylation doesn't happen
        else:
            # Late-stage (closer to 1.0) is better, scale to 0-10
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if the reaction involves N-alkylation by detecting:
        1. Formation of C-N bond where nitrogen was previously unsubstituted or less substituted
        2. Alkyl group attachment to nitrogen
        """
        try:
            rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles", "")
            if not rxn_smiles or ">>" not in rxn_smiles:
                return False
            
            reactants_smiles, product_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
            
            # Check for N-alkylation pattern
            return self._detect_n_alkylation(reactants, product)
            
        except Exception:
            return False
    
    def _detect_n_alkylation(self, reactants, product) -> bool:
        """
        Detect N-alkylation by checking for:
        1. Nitrogen atom that gains an alkyl substituent
        2. Carbon-nitrogen bond formation
        """
        # Get atom mappings for nitrogen atoms in product
        product_nitrogens = {}
        for atom in product.GetAtoms():
            if atom.GetAtomicNum() == 7 and atom.GetAtomMapNum() > 0:  # Nitrogen
                product_nitrogens[atom.GetAtomMapNum()] = atom
        
        # Check each nitrogen in reactants vs product
        for reactant in reactants:
            reactant_nitrogens = {}
            for atom in reactant.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetAtomMapNum() > 0:
                    reactant_nitrogens[atom.GetAtomMapNum()] = atom
            
            # Compare nitrogen substitution patterns
            for map_num, prod_n in product_nitrogens.items():
                if map_num in reactant_nitrogens:
                    react_n = reactant_nitrogens[map_num]
                    
                    # Count carbon neighbors (alkyl attachments)
                    react_carbons = sum(1 for neighbor in react_n.GetNeighbors() 
                                      if neighbor.GetAtomicNum() == 6)
                    prod_carbons = sum(1 for neighbor in prod_n.GetNeighbors() 
                                     if neighbor.GetAtomicNum() == 6)
                    
                    # N-alkylation: nitrogen gains carbon neighbor(s)
                    if prod_carbons > react_carbons:
                        # Additional check: ensure it's alkylation not arylation
                        if self._has_new_alkyl_attachment(react_n, prod_n, reactants, product):
                            return True
        
        return False
    
    def _has_new_alkyl_attachment(self, react_n, prod_n, reactants, product) -> bool:
        """
        Check if the new carbon attachment is an alkyl group (not aromatic)
        """
        react_carbon_maps = set()
        for neighbor in react_n.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                react_carbon_maps.add(neighbor.GetAtomMapNum())
        
        # Find new carbon attachments in product
        for neighbor in prod_n.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                if neighbor.GetAtomMapNum() not in react_carbon_maps:
                    # This is a new carbon attachment
                    # Check if it's aliphatic (alkyl) rather than aromatic
                    if not neighbor.GetIsAromatic():
                        return True
        
        return False
