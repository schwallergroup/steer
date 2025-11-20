"""Generated evaluation code for: Boc protecting group for amine alkylation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocAmineAlkylation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the use of Boc protecting groups to enable
    selective amine alkylation reactions. Checks for the sequence of Boc protection,
    alkylation, and deprotection steps.
    """
    
    def __init__(self, config):
        self.require_boc_protection = config.get("require_boc_protection", True)
        self.require_alkylation = config.get("require_alkylation", True)
        self.require_deprotection = config.get("require_deprotection", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        has_boc_protection = any(self.detect_boc_protection(r) for r in reactions)
        has_alkylation_on_boc = any(self.detect_alkylation_with_boc(r) for r in reactions)
        has_boc_deprotection = any(self.detect_boc_deprotection(r) for r in reactions)
        
        # Check if the sequence is present as required
        condition = True
        if self.require_boc_protection:
            condition = condition and has_boc_protection
        if self.require_alkylation:
            condition = condition and has_alkylation_on_boc
        if self.require_deprotection:
            condition = condition and has_boc_deprotection
            
        return condition, len(reactions)
    
    def detect_boc_protection(self, rxn):
        """Detect Boc protection of primary amine (R-NH2 + Boc2O -> R-NH-Boc)"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        # Look for Boc2O reagent in reactants
        boc2o_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
        has_boc2o = any(mol and mol.HasSubstructMatch(boc2o_pattern) for mol in reactant_mols if mol)
        
        # Look for primary amine in reactants and Boc-protected amine in products
        primary_amine_pattern = Chem.MolFromSmarts("[CH2,CH,C][NH2]")
        boc_amine_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
        
        has_primary_amine = any(mol and mol.HasSubstructMatch(primary_amine_pattern) 
                               for mol in reactant_mols if mol)
        has_boc_amine = any(mol and mol.HasSubstructMatch(boc_amine_pattern) 
                           for mol in product_mols if mol)
        
        return has_boc2o and has_primary_amine and has_boc_amine
    
    def detect_alkylation_with_boc(self, rxn):
        """Detect alkylation reaction where Boc-protected amine is present"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        # Look for Boc-protected amine in reactants
        boc_amine_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
        has_boc_amine_reactant = any(mol and mol.HasSubstructMatch(boc_amine_pattern) 
                                    for mol in reactant_mols if mol)
        
        # Look for alkyl halide or similar alkylating agent
        alkyl_halide_pattern = Chem.MolFromSmarts("[CH3,CH2,CH][Cl,Br,I]")
        tosylate_pattern = Chem.MolFromSmarts("[CH3,CH2,CH]OS(=O)(=O)c1ccc(C)cc1")
        mesylate_pattern = Chem.MolFromSmarts("[CH3,CH2,CH]OS(=O)(=O)C")
        
        has_alkylating_agent = any(
            mol and (mol.HasSubstructMatch(alkyl_halide_pattern) or 
                    mol.HasSubstructMatch(tosylate_pattern) or
                    mol.HasSubstructMatch(mesylate_pattern))
            for mol in reactant_mols if mol
        )
        
        # Check for increased molecular complexity (alkylation occurred)
        reactant_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in reactant_mols if mol)
        product_heavy_atoms = sum(mol.GetNumHeavyAtoms() for mol in product_mols if mol)
        
        return has_boc_amine_reactant and has_alkylating_agent and product_heavy_atoms > reactant_heavy_atoms - 5
    
    def detect_boc_deprotection(self, rxn):
        """Detect Boc deprotection (R-NH-Boc -> R-NH2 + CO2 + isobutene)"""
        reactants, products = rxn.split(">>")
        reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
        product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
        
        # Look for Boc-protected amine in reactants
        boc_amine_pattern = Chem.MolFromSmarts("CC(C)(C)OC(=O)N")
        has_boc_amine = any(mol and mol.HasSubstructMatch(boc_amine_pattern) 
                           for mol in reactant_mols if mol)
        
        # Look for free amine in products
        amine_pattern = Chem.MolFromSmarts("[NH2,NH]")
        has_free_amine = any(mol and mol.HasSubstructMatch(amine_pattern) 
                            for mol in product_mols if mol)
        
        # Look for typical deprotection conditions (TFA, HCl)
        reactant_smiles = reactants.lower()
        has_acid_conditions = any(acid in reactant_smiles for acid in ["tfa", "hcl", "cf3cooh", "trifluoroacetic"])
        
        return has_boc_amine and has_free_amine and (has_acid_conditions or len(product_mols) > len(reactant_mols))
