"""Generated evaluation code for: Boc protecting group cycling strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BocProtectingGroupCycling(MultiRxnCondBase):
    """
    Evaluates synthesis routes for Boc protecting group cycling strategy.
    Checks if Boc group is installed and then removed during the synthesis.
    Returns 1.0 if both operations are detected, 0.0 otherwise.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "Boc")
        self.required_operations = config.get("operations", ["installation", "removal"])
        
        # SMARTS patterns for Boc group detection
        self.boc_pattern = "NC(=O)OC(C)(C)C"  # Boc group attached to nitrogen
        self.free_amine_pattern = "[NH2,NH1]"  # Free amine patterns
    
    def condition_depth(self, d):
        """
        Check if both Boc installation and removal occur in the route.
        Returns (condition_met, total_reactions)
        """
        reactions = self.get_rxns(d)
        
        has_installation = any(self.detect_boc_installation(rxn) for rxn in reactions)
        has_removal = any(self.detect_boc_removal(rxn) for rxn in reactions)
        
        # Check if both required operations are present
        operations_found = []
        if has_installation:
            operations_found.append("installation")
        if has_removal:
            operations_found.append("removal")
        
        # Condition is met if all required operations are found
        condition_met = all(op in operations_found for op in self.required_operations)
        
        return condition_met, len(reactions)
    
    def detect_boc_installation(self, rxn):
        """
        Detect Boc group installation: free amine -> Boc-protected amine
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse molecules
        reactant_mols = []
        for r_smiles in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactant_mols.append(mol)
        
        product_mols = []
        for p_smiles in products.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is not None:
                product_mols.append(mol)
        
        # Check if reactants have free amine and products have Boc group
        boc_pattern_mol = Chem.MolFromSmarts(self.boc_pattern)
        amine_pattern_mol = Chem.MolFromSmarts(self.free_amine_pattern)
        
        # Reactants should have free amine, products should have Boc
        reactants_have_amine = any(mol.HasSubstructMatch(amine_pattern_mol) for mol in reactant_mols)
        products_have_boc = any(mol.HasSubstructMatch(boc_pattern_mol) for mol in product_mols)
        reactants_have_boc = any(mol.HasSubstructMatch(boc_pattern_mol) for mol in reactant_mols)
        
        # Installation: amine present in reactants, Boc present in products, no Boc in reactants
        return reactants_have_amine and products_have_boc and not reactants_have_boc
    
    def detect_boc_removal(self, rxn):
        """
        Detect Boc group removal: Boc-protected amine -> free amine
        """
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
        
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        # Parse molecules
        reactant_mols = []
        for r_smiles in reactants.split("."):
            mol = Chem.MolFromSmiles(r_smiles)
            if mol is not None:
                reactant_mols.append(mol)
        
        product_mols = []
        for p_smiles in products.split("."):
            mol = Chem.MolFromSmiles(p_smiles)
            if mol is not None:
                product_mols.append(mol)
        
        # Check if reactants have Boc group and products have free amine
        boc_pattern_mol = Chem.MolFromSmarts(self.boc_pattern)
        amine_pattern_mol = Chem.MolFromSmarts(self.free_amine_pattern)
        
        reactants_have_boc = any(mol.HasSubstructMatch(boc_pattern_mol) for mol in reactant_mols)
        products_have_amine = any(mol.HasSubstructMatch(amine_pattern_mol) for mol in product_mols)
        products_have_boc = any(mol.HasSubstructMatch(boc_pattern_mol) for mol in product_mols)
        
        # Removal: Boc present in reactants, amine present in products, no Boc in products
        return reactants_have_boc and products_have_amine and not products_have_boc
