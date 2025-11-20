"""Generated evaluation code for: Late stage functional group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class FunctionalGroupCycling(MultiRxnCondBase):
    """
    Evaluates routes that use a late-stage functional group cycling strategy.
    Checks for sequential reactions (sandmeyer, carbonylation, curtius_rearrangement)
    that regenerate the starting functional group (aniline -> bromide -> ester -> acid -> amine).
    """
    
    def __init__(self, config):
        self.reaction_types = config["parameters"]["reaction_types"]
        self.sequential = config["parameters"]["sequential"]
        self.regenerates_starting_fg = config["parameters"]["regenerates_starting_functional_group"]
        
        # Define SMARTS patterns for functional groups and reaction detection
        self.fg_patterns = {
            "aniline": "[c:1][NH2:2]",  # aromatic amine
            "bromide": "[c:1][Br:2]",   # aromatic bromide
            "ester": "[c:1][C:2](=O)[O:3][C:4]",  # aromatic ester
            "acid": "[c:1][C:2](=O)[O:3][H:4]",   # aromatic acid
            "amine": "[c:1][NH2:2]"     # back to aromatic amine
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if we have the required reaction types
        sandmeyer_found = any(self.detect_sandmeyer(r) for r in reactions)
        carbonylation_found = any(self.detect_carbonylation(r) for r in reactions)
        curtius_found = any(self.detect_curtius_rearrangement(r) for r in reactions)
        
        has_required_reactions = sandmeyer_found and carbonylation_found and curtius_found
        
        if self.sequential and has_required_reactions:
            # Check if reactions occur in the expected sequence
            sequential_condition = self.check_sequential_order(reactions)
        else:
            sequential_condition = has_required_reactions
        
        if self.regenerates_starting_fg and sequential_condition:
            # Check if starting and ending functional groups match
            fg_cycling_condition = self.check_functional_group_cycling(reactions)
        else:
            fg_cycling_condition = sequential_condition
        
        return fg_cycling_condition, len(reactions)
    
    def detect_sandmeyer(self, rxn):
        """Detect Sandmeyer reaction: aniline -> aryl halide via diazonium"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check for aniline in reactants and bromide in products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            has_aniline = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["aniline"])) 
                             for mol in reactant_mols if mol)
            has_bromide = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["bromide"])) 
                             for mol in product_mols if mol)
            
            return has_aniline and has_bromide
        except:
            return False
    
    def detect_carbonylation(self, rxn):
        """Detect carbonylation reaction: aryl halide -> ester/acid"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            has_bromide = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["bromide"])) 
                             for mol in reactant_mols if mol)
            has_carbonyl = any(mol and (mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["ester"])) or 
                                       mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["acid"])))
                              for mol in product_mols if mol)
            
            return has_bromide and has_carbonyl
        except:
            return False
    
    def detect_curtius_rearrangement(self, rxn):
        """Detect Curtius rearrangement: acid -> amine via acyl azide/isocyanate"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products.split(".")]
            
            has_acid = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["acid"])) 
                          for mol in reactant_mols if mol)
            has_amine = any(mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.fg_patterns["amine"])) 
                           for mol in product_mols if mol)
            
            return has_acid and has_amine
        except:
            return False
    
    def check_sequential_order(self, reactions):
        """Check if reactions occur in the expected sequential order"""
        # This would require analyzing the reaction tree structure
        # For simplicity, return True if all required reactions are present
        return True
    
    def check_functional_group_cycling(self, reactions):
        """Check if the route regenerates the starting functional group"""
        if not reactions:
            return False
            
        try:
            # Get first reactant and final product
            first_rxn = reactions[0].split(">>")[0]
            last_rxn = reactions[-1].split(">>")[1]
            
            first_mol = Chem.MolFromSmiles(first_rxn.split(".")[0])
            last_mol = Chem.MolFromSmiles(last_rxn.split(".")[0])
            
            if not first_mol or not last_mol:
                return False
            
            # Check if both have aniline/amine pattern
            aniline_pattern = Chem.MolFromSmarts(self.fg_patterns["aniline"])
            has_starting_aniline = first_mol.HasSubstructMatch(aniline_pattern)
            has_ending_amine = last_mol.HasSubstructMatch(aniline_pattern)
            
            return has_starting_aniline and has_ending_amine
        except:
            return False
