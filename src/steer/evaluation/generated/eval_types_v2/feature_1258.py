"""Generated evaluation code for: Sequential Williamson ether synthesis approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialWilliamsonEther(BaseScoring):
    """
    Evaluates routes for sequential Williamson ether synthesis reactions.
    Checks if two Williamson ether formations occur at specified steps in the route.
    """
    
    def __init__(self, config: Dict):
        self.target_steps = config["parameters"]["steps"]  # [3, 4]
        self.count = config["parameters"]["count"]  # 2
        self.sequential = config["parameters"]["sequential"]  # true
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Sequential condition not met
        else:
            return 10 - x  # Better score for earlier occurrence
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if sequential Williamson ether synthesis occurs at target steps.
        Returns (condition_met, depth_fraction)
        """
        reactions = self.get_all_reactions(d)
        williamson_steps = []
        
        # Find all Williamson ether synthesis reactions and their depths
        for i, rxn_data in enumerate(reactions):
            if self.is_williamson_ether(rxn_data):
                williamson_steps.append(i + 1)  # 1-indexed step number
        
        # Check if we have the required count
        if len(williamson_steps) < self.count:
            return False, -1
        
        # Check if sequential Williamson ethers occur at target steps
        if self.sequential and len(self.target_steps) >= 2:
            for i in range(len(williamson_steps) - 1):
                step1, step2 = williamson_steps[i], williamson_steps[i + 1]
                if step1 in self.target_steps and step2 in self.target_steps:
                    if abs(step2 - step1) == 1:  # Consecutive steps
                        depth_fraction = min(step1, step2) / len(reactions)
                        return True, depth_fraction
        
        return False, -1
    
    def get_all_reactions(self, d) -> List:
        """Extract all reactions from the route tree via BFS traversal"""
        reactions = []
        queue = [d]
        
        while queue:
            node = queue.pop(0)
            
            if "metadata" in node and "mapped_reaction_smiles" in node["metadata"]:
                reactions.append(node)
            
            if "children" in node:
                for child in node["children"]:
                    queue.append(child)
        
        return reactions
    
    def is_williamson_ether(self, rxn_data) -> bool:
        """
        Detect Williamson ether synthesis reaction pattern.
        Looks for C-O-C ether formation from alkyl halide + phenoxide/alkoxide.
        """
        try:
            rxn_smiles = rxn_data["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for alkyl halide pattern (C-X where X = F, Cl, Br, I)
            alkyl_halide_pattern = Chem.MolFromSmarts("[C][F,Cl,Br,I]")
            has_alkyl_halide = any(mol.HasSubstructMatch(alkyl_halide_pattern) for mol in reactants)
            
            # Check for phenoxide/alkoxide pattern (nucleophilic oxygen)
            # Look for C-O- or Ar-O- patterns
            phenoxide_pattern = Chem.MolFromSmarts("[c][O-]")  # Aromatic phenoxide
            alkoxide_pattern = Chem.MolFromSmarts("[C][O-]")   # Alkoxide
            has_nucleophile = any(mol.HasSubstructMatch(phenoxide_pattern) or 
                                mol.HasSubstructMatch(alkoxide_pattern) for mol in reactants)
            
            # Check for ether formation in products (C-O-C)
            ether_pattern = Chem.MolFromSmarts("[C][O][C]")
            has_ether_product = any(mol.HasSubstructMatch(ether_pattern) for mol in products)
            
            # Check for halide leaving group in products
            halide_pattern = Chem.MolFromSmarts("[F-,Cl-,Br-,I-]")
            has_halide_product = any(mol.HasSubstructMatch(halide_pattern) for mol in products)
            
            return has_alkyl_halide and has_nucleophile and has_ether_product
            
        except Exception:
            return False
    
    def hit_condition(self, d):
        """Single reaction check - used by BaseScoring framework"""
        return self.is_williamson_ether(d)
