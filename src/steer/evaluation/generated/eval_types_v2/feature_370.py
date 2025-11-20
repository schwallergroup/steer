"""Generated evaluation code for: Late stage N-alkylation functionalization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNAlkylation(BaseScoring):
    """
    Evaluates whether N-alkylation occurs in the late stage of synthesis.
    Returns higher scores when N-alkylation happens after the stage threshold.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config["parameters"].get("stage_threshold", 0.8)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # N-alkylation doesn't occur
        
        # x is the depth fraction where N-alkylation occurs
        # Higher scores for later stage reactions (closer to 1.0)
        if x >= self.stage_threshold:
            return 10  # Perfect score for late-stage N-alkylation
        else:
            # Penalize early N-alkylation, with linear scaling
            return max(0, 10 * (x / self.stage_threshold))
    
    def hit_condition(self, d) -> bool:
        """
        Detects N-alkylation reactions by looking for:
        1. Formation of C-N bonds where nitrogen gains an alkyl group
        2. Nitrogen atom that increases its substitution degree
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not all(reactants):
                return False
            
            # Get all nitrogen atoms in product with their map numbers
            product_nitrogens = {}
            for atom in product.GetAtoms():
                if atom.GetAtomicNum() == 7 and atom.GetAtomMapNum() > 0:  # Nitrogen
                    map_num = atom.GetAtomMapNum()
                    # Count carbon neighbors (alkyl groups)
                    carbon_neighbors = sum(1 for neighbor in atom.GetNeighbors() 
                                         if neighbor.GetAtomicNum() == 6)
                    product_nitrogens[map_num] = carbon_neighbors
            
            # Get nitrogen substitution in reactants
            reactant_nitrogens = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomicNum() == 7 and atom.GetAtomMapNum() > 0:
                        map_num = atom.GetAtomMapNum()
                        carbon_neighbors = sum(1 for neighbor in atom.GetNeighbors() 
                                             if neighbor.GetAtomicNum() == 6)
                        reactant_nitrogens[map_num] = carbon_neighbors
            
            # Check if any nitrogen gained carbon substituents (N-alkylation)
            for map_num, product_carbons in product_nitrogens.items():
                reactant_carbons = reactant_nitrogens.get(map_num, 0)
                if product_carbons > reactant_carbons:
                    # Additional check: ensure we're forming C-N bonds, not just rearranging
                    if self._involves_alkyl_transfer(reactants, product, map_num):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _involves_alkyl_transfer(self, reactants, product, nitrogen_map_num):
        """
        Verify that an alkyl group is being transferred to nitrogen,
        typical of N-alkylation reactions.
        """
        try:
            # Look for common N-alkylation patterns:
            # 1. Alkyl halides + amines
            # 2. Alcohols + amines (reductive amination)
            # 3. Alkyl tosylates/mesylates + amines
            
            # Check for leaving groups in reactants (halides, tosylates, etc.)
            leaving_group_patterns = [
                "[C][Cl,Br,I]",  # Alkyl halides
                "[C][O][S](=O)(=O)[c]",  # Tosylates
                "[C][O][S](=O)(=O)[CH3]",  # Mesylates
            ]
            
            has_alkylating_agent = False
            for reactant in reactants:
                for pattern in leaving_group_patterns:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        has_alkylating_agent = True
                        break
                if has_alkylating_agent:
                    break
            
            # Also check for reductive amination (carbonyl + amine -> alkylated amine)
            if not has_alkylating_agent:
                carbonyl_pattern = "[C]=[O]"
                for reactant in reactants:
                    if reactant.HasSubstructMatch(Chem.MolFromSmarts(carbonyl_pattern)):
                        has_alkylating_agent = True
                        break
            
            return has_alkylating_agent
            
        except Exception:
            return True  # Default to True if pattern matching fails
