"""Generated evaluation code for: Convergent synthesis via reagent preparation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentReagentPrep(MultiRxnCondBase):
    """
    Evaluates convergent synthesis strategy involving reagent preparation for double alkylation.
    Checks for preparation of N-nosyl-bis(2-chloroethyl)amine or similar bis-electrophilic reagents
    and their use in piperazine formation via double alkylation.
    """
    
    def __init__(self, config):
        self.fragment_count = config.get("fragment_count", 2)
        self.coupling_reaction = config.get("coupling_reaction", "double_alkylation")
        self.require_reagent_prep = config.get("reagent_preparation", True)
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check for reagent preparation step
        reagent_prep_found = any(self.detect_reagent_preparation(r) for r in reactions)
        
        # Check for double alkylation/coupling reaction
        coupling_found = any(self.detect_double_alkylation(r) for r in reactions)
        
        # Check convergent strategy - multiple fragments coming together
        convergent_strategy = self.assess_convergence(reactions)
        
        condition_met = (
            reagent_prep_found == self.require_reagent_prep and
            coupling_found and
            convergent_strategy
        )
        
        return condition_met, len(reactions)
    
    def detect_reagent_preparation(self, rxn):
        """Detect preparation of bis-electrophilic reagents like N-nosyl-bis(2-chloroethyl)amine"""
        # Pattern for N-nosyl-bis(2-chloroethyl)amine
        nosyl_pattern = "[N+](=O)([O-])c1ccc(S(=O)(=O)N(CCCl)CCCl)cc1"
        
        # More general bis-alkyl halide patterns
        bis_chloroethyl_patterns = [
            "ClCCN(CCCl)S(=O)(=O)",  # Generic bis-chloroethyl sulfonamide
            "ClCCN(CCCl)C(=O)",      # Bis-chloroethyl amide
            "ClCCN(CCCl)[CH2]"       # General bis-chloroethyl amine
        ]
        
        products = rxn.split(">>")[0].split(".")
        
        for prod_smiles in products:
            try:
                mol = Chem.MolFromSmiles(prod_smiles)
                if mol is None:
                    continue
                
                # Check for nosyl pattern
                nosyl_mol = Chem.MolFromSmarts(nosyl_pattern)
                if nosyl_mol and mol.HasSubstructMatch(nosyl_mol):
                    return True
                
                # Check for general bis-electrophile patterns
                for pattern in bis_chloroethyl_patterns:
                    pattern_mol = Chem.MolFromSmarts(pattern)
                    if pattern_mol and mol.HasSubstructMatch(pattern_mol):
                        return True
                        
            except:
                continue
                
        return False
    
    def detect_double_alkylation(self, rxn):
        """Detect double alkylation reactions leading to piperazine formation"""
        # Piperazine core pattern
        piperazine_pattern = "N1CCNCC1"
        
        # Bis-alkylation patterns (two alkyl chains attached to nitrogens)
        alkylated_piperazine_patterns = [
            "N1([CH2])CCN([CH2])CC1",  # Basic substituted piperazine
            "N1(C)CCN(C)CC1"           # Alkyl-substituted piperazine
        ]
        
        try:
            products = rxn.split(">>")[0].split(".")
            reactants = rxn.split(">>")[1].split(".")
            
            # Check if product contains piperazine core
            piperazine_in_product = False
            for prod_smiles in products:
                mol = Chem.MolFromSmiles(prod_smiles)
                if mol is None:
                    continue
                    
                pip_mol = Chem.MolFromSmarts(piperazine_pattern)
                if pip_mol and mol.HasSubstructMatch(pip_mol):
                    piperazine_in_product = True
                    break
            
            # Check for bis-electrophile in reactants
            bis_electrophile_in_reactants = False
            for react_smiles in reactants:
                mol = Chem.MolFromSmiles(react_smiles)
                if mol is None:
                    continue
                    
                # Look for molecules with two leaving groups (Cl, Br, I)
                halogen_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() in ['Cl', 'Br', 'I']:
                        halogen_count += 1
                
                if halogen_count >= 2:
                    bis_electrophile_in_reactants = True
                    break
            
            return piperazine_in_product and bis_electrophile_in_reactants
            
        except:
            return False
    
    def assess_convergence(self, reactions):
        """Assess if the synthesis follows a convergent strategy"""
        if len(reactions) < 2:
            return False
            
        # Look for reactions that combine multiple fragments
        convergent_reactions = 0
        
        for rxn in reactions:
            try:
                reactants = rxn.split(">>")[1].split(".")
                products = rxn.split(">>")[0].split(".")
                
                # Convergent step typically has multiple reactants forming fewer products
                if len(reactants) >= self.fragment_count and len(products) <= len(reactants):
                    convergent_reactions += 1
                    
            except:
                continue
                
        return convergent_reactions >= 1
