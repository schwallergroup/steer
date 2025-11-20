"""Generated evaluation code for: Sequential halogen exchange via alcohol intermediate"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialHalogenExchange(MultiRxnCondBase):
    """
    Detects inefficient sequential halogen exchange via alcohol intermediate.
    Identifies routes that convert C-Br to C-Cl through C-OH intermediate
    instead of using direct halogen exchange.
    """
    
    def __init__(self, config):
        self.detect_sequence = config.get("detect_sequence", True)
        # SMARTS patterns for detecting transformations
        self.br_to_oh_pattern = "[#6]-[Br]>>[#6]-[OH]"  # C-Br to C-OH
        self.oh_to_cl_pattern = "[#6]-[OH]>>[#6]-[Cl]"  # C-OH to C-Cl
        self.direct_br_to_cl_pattern = "[#6]-[Br]>>[#6]-[Cl]"  # Direct C-Br to C-Cl
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Look for the sequence: Br->OH->Cl
        sequence_found = self.detect_halogen_exchange_sequence(reactions)
        
        condition = sequence_found == self.detect_sequence
        return condition, len(reactions)
    
    def detect_halogen_exchange_sequence(self, reactions) -> bool:
        """
        Detect if the route contains C-Br -> C-OH -> C-Cl sequence
        """
        br_to_oh_reactions = []
        oh_to_cl_reactions = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_br_to_oh_transformation(rxn):
                br_to_oh_reactions.append(i)
            elif self.detect_oh_to_cl_transformation(rxn):
                oh_to_cl_reactions.append(i)
        
        # Check if we have both transformations and they form a sequence
        if br_to_oh_reactions and oh_to_cl_reactions:
            return self.verify_sequence_connectivity(reactions, br_to_oh_reactions, oh_to_cl_reactions)
        
        return False
    
    def detect_br_to_oh_transformation(self, rxn) -> bool:
        """Detect C-Br to C-OH transformation"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for loss of Br and gain of OH on same carbon
            br_pattern = Chem.MolFromSmarts("[#6]-[Br]")
            oh_pattern = Chem.MolFromSmarts("[#6]-[OH]")
            
            has_br_reactant = any(mol.HasSubstructMatch(br_pattern) for mol in reactant_mols if mol)
            has_oh_product = any(mol.HasSubstructMatch(oh_pattern) for mol in product_mols if mol)
            
            return has_br_reactant and has_oh_product
        except:
            return False
    
    def detect_oh_to_cl_transformation(self, rxn) -> bool:
        """Detect C-OH to C-Cl transformation"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for loss of OH and gain of Cl on same carbon
            oh_pattern = Chem.MolFromSmarts("[#6]-[OH]")
            cl_pattern = Chem.MolFromSmarts("[#6]-[Cl]")
            
            has_oh_reactant = any(mol.HasSubstructMatch(oh_pattern) for mol in reactant_mols if mol)
            has_cl_product = any(mol.HasSubstructMatch(cl_pattern) for mol in product_mols if mol)
            
            return has_oh_reactant and has_cl_product
        except:
            return False
    
    def verify_sequence_connectivity(self, reactions, br_to_oh_indices, oh_to_cl_indices) -> bool:
        """
        Verify that the Br->OH and OH->Cl transformations are connected
        (i.e., the OH intermediate from first reaction is used in second reaction)
        """
        # For each Br->OH reaction, check if its product feeds into an OH->Cl reaction
        for br_idx in br_to_oh_indices:
            for cl_idx in oh_to_cl_indices:
                # In a synthesis tree, later reactions (higher index) use products from earlier ones
                if br_idx < cl_idx:
                    # Simple heuristic: if we found both transformations in sequence, 
                    # assume they're connected (more sophisticated analysis would require
                    # tracking specific molecules through the synthesis tree)
                    return True
        
        return False
