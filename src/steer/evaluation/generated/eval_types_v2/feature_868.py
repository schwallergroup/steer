"""Generated evaluation code for: Late stage Grignard methyl addition"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageGrignardMethylAddition(BaseScoring):
    """
    Evaluates whether a Grignard methyl addition reaction occurs late in the synthesis route.
    
    Detects the formation of C-C bonds where a methyl group is added to a carbonyl
    (aldehyde or ketone) via Grignard reaction, and scores based on how late in the
    synthesis this occurs.
    """
    
    def __init__(self, config: Dict):
        self.depth_threshold = config.get("depth_threshold", 3)
        self.timing = config.get("timing", "late")
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Grignard addition doesn't happen
        
        if self.timing == "late":
            # Reward later stage reactions (higher depth fraction)
            return x * 10  # Convert to 0-10 scale
        else:
            # Reward earlier stage reactions (lower depth fraction)
            return (1 - x) * 10
    
    def hit_condition(self, d) -> bool:
        """
        Check if this reaction node represents a Grignard methyl addition.
        
        Detects:
        1. Formation of C-C bond between methyl group and former carbonyl carbon
        2. Conversion of C=O to C-OH
        3. Addition pattern consistent with Grignard mechanism
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        try:
            rxn_parts = rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product = Chem.MolFromSmiles(rxn_parts[0])
            reactants = [Chem.MolFromSmiles(r) for r in rxn_parts[1].split(".")]
            
            if not product or not all(reactants):
                return False
                
            return self._detect_grignard_methyl_addition(product, reactants)
            
        except:
            return False
    
    def _detect_grignard_methyl_addition(self, product, reactants):
        """
        Detect Grignard methyl addition by checking:
        1. Product has new C-OH group with adjacent methyl
        2. Reactants contain carbonyl and methyl source
        """
        # Pattern for newly formed tertiary or secondary alcohol with methyl
        # [CH3]-C-[OH] where C was previously part of C=O
        alcohol_methyl_pattern = Chem.MolFromSmarts("[CH3]-[CH1,CH0]-[OH]")
        
        if not product.HasSubstructMatch(alcohol_methyl_pattern):
            return False
            
        # Check reactants for carbonyl pattern
        carbonyl_pattern = Chem.MolFromSmarts("[CH1,CH0]=[OH0]")
        has_carbonyl = any(r.HasSubstructMatch(carbonyl_pattern) for r in reactants)
        
        if not has_carbonyl:
            return False
            
        # Look for methyl addition by comparing mapped atoms
        return self._verify_methyl_addition_mapping(product, reactants)
    
    def _verify_methyl_addition_mapping(self, product, reactants):
        """
        Verify methyl addition by checking atom mapping numbers.
        """
        try:
            # Get mapped atoms in product
            prod_map_to_atom = {}
            for atom in product.GetAtoms():
                if atom.GetAtomMapNum() > 0:
                    prod_map_to_atom[atom.GetAtomMapNum()] = atom
            
            # Get mapped atoms in reactants
            react_map_to_atom = {}
            for reactant in reactants:
                for atom in reactant.GetAtoms():
                    if atom.GetAtomMapNum() > 0:
                        react_map_to_atom[atom.GetAtomMapNum()] = atom
            
            # Look for carbon that changed from C=O to C-OH with new methyl neighbor
            for map_num, prod_atom in prod_map_to_atom.items():
                if map_num in react_map_to_atom:
                    react_atom = react_map_to_atom[map_num]
                    
                    # Check if this carbon went from sp2 to sp3 (C=O to C-OH)
                    if (react_atom.GetHybridization() == Chem.HybridizationType.SP2 and
                        prod_atom.GetHybridization() == Chem.HybridizationType.SP3):
                        
                        # Check if product has new methyl neighbor not in reactants
                        if self._has_new_methyl_neighbor(prod_atom, react_atom, prod_map_to_atom, react_map_to_atom):
                            return True
            
            return False
        except:
            return False
    
    def _has_new_methyl_neighbor(self, prod_atom, react_atom, prod_map_to_atom, react_map_to_atom):
        """
        Check if the product atom has a new methyl neighbor that wasn't present in reactants.
        """
        # Get neighbors in product
        prod_neighbors = set()
        for neighbor in prod_atom.GetNeighbors():
            if neighbor.GetAtomMapNum() > 0:
                prod_neighbors.add(neighbor.GetAtomMapNum())
        
        # Get neighbors in reactant
        react_neighbors = set()
        for neighbor in react_atom.GetNeighbors():
            if neighbor.GetAtomMapNum() > 0:
                react_neighbors.add(neighbor.GetAtomMapNum())
        
        # Find new neighbors
        new_neighbors = prod_neighbors - react_neighbors
        
        # Check if any new neighbor is a methyl carbon
        for new_map_num in new_neighbors:
            if new_map_num in prod_map_to_atom:
                new_atom = prod_map_to_atom[new_map_num]
                if (new_atom.GetSymbol() == 'C' and 
                    new_atom.GetDegree() == 1 and
                    sum(1 for n in new_atom.GetNeighbors() if n.GetSymbol() == 'H') == 3):
                    return True
        
        return False
