"""Generated evaluation code for: Sequential benzyl protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialBenzylProtection(MultiRxnCondBase):
    """
    Evaluates whether benzyl protecting groups are removed in sequential steps
    rather than simultaneously. Checks for separate deprotection reactions
    of benzyl ethers and benzyl esters.
    """
    
    def __init__(self, config):
        self.protecting_group = config["parameters"]["protecting_group"]
        self.count = config["parameters"]["count"]
        self.sequential = config["parameters"]["sequential"]
        
        # Define SMARTS patterns for benzyl protecting groups
        self.benzyl_ether_pattern = "[CH2]c1ccccc1-[OH]"  # Benzyl ether being formed/broken
        self.benzyl_ester_pattern = "[CH2]c1ccccc1-[O][C]=O"  # Benzyl ester being formed/broken
        self.benzyl_general_pattern = "[CH2]c1ccccc1"  # General benzyl group

    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find all benzyl deprotection reactions
        benzyl_deprotections = []
        for i, rxn in enumerate(reactions):
            if self.is_benzyl_deprotection(rxn):
                deprotection_type = self.get_deprotection_type(rxn)
                benzyl_deprotections.append((i, deprotection_type))
        
        # Check if we have the required number of deprotections
        if len(benzyl_deprotections) < self.count:
            return False, len(reactions)
        
        # Check if deprotections are sequential (different types in separate reactions)
        if self.sequential:
            deprotection_types = [dep_type for _, dep_type in benzyl_deprotections]
            # Should have different types and not occur in same reaction
            unique_types = set(deprotection_types)
            reaction_indices = [idx for idx, _ in benzyl_deprotections]
            
            condition = (len(unique_types) >= 2 and 
                        len(set(reaction_indices)) == len(benzyl_deprotections))
        else:
            condition = len(benzyl_deprotections) >= self.count
            
        return condition, len(reactions)

    def is_benzyl_deprotection(self, rxn):
        """Check if reaction involves benzyl group removal"""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Check if benzyl group is present in reactants
            reactant_mol = Chem.MolFromSmiles(reactants)
            if not reactant_mol:
                return False
                
            has_benzyl_reactant = reactant_mol.HasSubstructMatch(
                Chem.MolFromSmarts(self.benzyl_general_pattern)
            )
            
            if not has_benzyl_reactant:
                return False
            
            # Check if benzyl group appears as separate product (indicating cleavage)
            benzyl_products = []
            for prod_smiles in products:
                prod_mol = Chem.MolFromSmiles(prod_smiles)
                if prod_mol and prod_mol.HasSubstructMatch(
                    Chem.MolFromSmarts(self.benzyl_general_pattern)
                ):
                    benzyl_products.append(prod_smiles)
            
            # Deprotection if benzyl group cleaved off as separate product
            return len(benzyl_products) > 0
            
        except:
            return False

    def get_deprotection_type(self, rxn):
        """Determine if deprotection is of ether or ester type"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0]
            reactant_mol = Chem.MolFromSmiles(reactants)
            
            if not reactant_mol:
                return "unknown"
            
            if reactant_mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)):
                return "ether"
            elif reactant_mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ester_pattern)):
                return "ester"
            else:
                return "other"
                
        except:
            return "unknown"

    def route_scoring(self, condition_met, total_reactions):
        """Score based on whether sequential deprotection strategy is used"""
        if condition_met:
            return 10.0  # Perfect score for meeting sequential strategy
        else:
            return 0.0   # No points if strategy not followed
