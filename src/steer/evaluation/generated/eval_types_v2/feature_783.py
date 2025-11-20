"""Generated evaluation code for: Multi-step nitro to carboxylic acid conversion"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultiStepNitroToCarboxylicAcid(MultiRxnCondBase):
    """
    Evaluates routes for multi-step nitro to carboxylic acid conversion.
    Checks if the route follows an unusually long 5-step pathway through
    nitro -> amine -> bromide -> nitrile -> ketone -> carboxylic acid
    rather than using direct conversion methods.
    """
    
    def __init__(self, config):
        self.start_group = config["parameters"]["start_group"]
        self.end_group = config["parameters"]["end_group"]
        self.min_steps = config["parameters"]["min_steps"]
        self.pathway = config["parameters"]["pathway"]
        
        # Compile SMARTS patterns
        self.patterns = [Chem.MolFromSmarts(smarts) for smarts in self.pathway]
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the route follows the multi-step nitro to carboxylic acid pathway"""
        reactions = self.get_rxns(d)
        
        # Find if we have the start and end groups in the route
        has_start_group = False
        has_end_group = False
        pathway_steps = []
        
        for rxn in reactions:
            rxn_smiles = rxn["metadata"]["mapped_reaction_smiles"].split(">>")
            products = [Chem.MolFromSmiles(p) for p in rxn_smiles[0].split(".")]
            reactants = [Chem.MolFromSmiles(r) for r in rxn_smiles[1].split(".")]
            
            # Check all molecules in this reaction step
            all_mols = products + reactants
            
            for i, pattern in enumerate(self.patterns):
                for mol in all_mols:
                    if mol and mol.HasSubstructMatch(pattern):
                        if i == 0:  # Start group (nitro)
                            has_start_group = True
                        elif i == len(self.patterns) - 1:  # End group (carboxylic acid)
                            has_end_group = True
                        
                        if i not in pathway_steps:
                            pathway_steps.append(i)
        
        # Check if we have the required pathway transformation
        pathway_steps.sort()
        
        # Must have start and end groups, and follow sequential pathway
        condition = (has_start_group and 
                    has_end_group and 
                    len(pathway_steps) >= self.min_steps and
                    self._follows_sequential_pathway(pathway_steps))
        
        return condition, len(reactions)
    
    def _follows_sequential_pathway(self, found_steps):
        """Check if the found steps follow a reasonable sequential pathway"""
        # Must include start (0) and end (last index) 
        if 0 not in found_steps or (len(self.patterns) - 1) not in found_steps:
            return False
            
        # Should have at least min_steps intermediate transformations
        if len(found_steps) < self.min_steps:
            return False
            
        # Check for reasonable progression (not necessarily all steps)
        # but should have key intermediates like amine (1), nitrile (3)
        key_intermediates = [1, 3]  # amine and nitrile are key steps
        has_key_intermediates = any(step in found_steps for step in key_intermediates)
        
        return has_key_intermediates
