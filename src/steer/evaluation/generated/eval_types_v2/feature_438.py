"""Generated evaluation code for: Ethyl to methyl ester conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EthylToMethylEsterConversion(MultiRxnCondBase):
    """
    Checks if the route contains saponification followed by methylation
    to convert ethyl esters to methyl esters.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.ethyl_ester_pattern = Chem.MolFromSmarts("[CH3][CH2]OC(=O)")
        self.methyl_ester_pattern = Chem.MolFromSmarts("[CH3]OC(=O)")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        saponification_found = False
        methylation_found = False
        saponification_depth = -1
        methylation_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_saponification(rxn):
                saponification_found = True
                saponification_depth = i
                
            if self.detect_methylation(rxn):
                methylation_found = True
                methylation_depth = i
        
        # Check if both reactions are present
        both_present = saponification_found and methylation_found
        
        # Check if they occur in the correct sequence (saponification before methylation)
        correct_sequence = saponification_depth < methylation_depth if both_present else False
        
        if self.require_sequence:
            condition = both_present and correct_sequence
        else:
            condition = both_present
            
        return condition, len(reactions)
    
    def detect_saponification(self, rxn):
        """Detect saponification: ethyl ester -> carboxylic acid + ethanol"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        # Check if reactants contain ethyl ester
        ethyl_ester_in_reactants = any(
            mol and mol.HasSubstructMatch(self.ethyl_ester_pattern) 
            for mol in reactants
        )
        
        # Check if products contain carboxylic acid
        carboxylic_acid_in_products = any(
            mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern)
            for mol in products
        )
        
        # Check if ethanol is produced
        ethanol_pattern = Chem.MolFromSmarts("CCO")
        ethanol_in_products = any(
            mol and mol.HasSubstructMatch(ethanol_pattern)
            for mol in products
        )
        
        return ethyl_ester_in_reactants and carboxylic_acid_in_products and ethanol_in_products
    
    def detect_methylation(self, rxn):
        """Detect methylation: carboxylic acid -> methyl ester"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = [Chem.MolFromSmiles(r.strip()) for r in rxn_parts[0].split(".")]
        products = [Chem.MolFromSmiles(p.strip()) for p in rxn_parts[1].split(".")]
        
        # Check if reactants contain carboxylic acid
        carboxylic_acid_in_reactants = any(
            mol and mol.HasSubstructMatch(self.carboxylic_acid_pattern)
            for mol in reactants
        )
        
        # Check if products contain methyl ester
        methyl_ester_in_products = any(
            mol and mol.HasSubstructMatch(self.methyl_ester_pattern)
            for mol in products
        )
        
        # Check for methylating agent in reactants (common ones)
        methylating_agents = [
            Chem.MolFromSmarts("COC"),  # dimethyl carbonate
            Chem.MolFromSmarts("CO"),   # methanol
            Chem.MolFromSmarts("CI"),   # methyl iodide
        ]
        
        methylating_agent_present = any(
            agent and any(mol and mol.HasSubstructMatch(agent) for mol in reactants)
            for agent in methylating_agents
        )
        
        return carboxylic_acid_in_reactants and methyl_ester_in_products and methylating_agent_present
