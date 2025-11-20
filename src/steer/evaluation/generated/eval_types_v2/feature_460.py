"""Generated evaluation code for: Non-orthogonal protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class NonOrthogonalProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether a synthesis route uses non-orthogonal protecting group strategy.
    Checks for simultaneous presence of protecting groups that require identical 
    deprotection conditions (e.g., tert-butyl ester and Boc both being acid-labile).
    """
    
    def __init__(self, config):
        self.protecting_groups = config.get("protecting_groups", ["tert-butyl ester", "Boc"])
        self.orthogonality = config.get("orthogonality", "non-orthogonal")
        self.deprotection_conditions = config.get("deprotection_conditions", "both acid-labile")
        
        # Define SMARTS patterns for protecting groups
        self.pg_patterns = {
            "tert-butyl ester": "[CX3](=O)[OX2]C(C)(C)C",
            "Boc": "[NX3][CX3](=O)[OX2]C(C)(C)C",
            "Cbz": "[NX3][CX3](=O)[OX2][CH2]c1ccccc1",
            "Fmoc": "[NX3][CX3](=O)[OX2][CH2][CH]1c2ccccc2-c2ccccc12",
            "benzyl ester": "[CX3](=O)[OX2][CH2]c1ccccc1",
            "methyl ester": "[CX3](=O)[OX2]C",
            "TBS": "[SiX4]([CH3])([CH3])C(C)(C)C",
            "TBDPS": "[SiX4](c1ccccc1)(c2ccccc2)C(C)(C)C"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Check if target protecting groups are present in any molecule in the route
        pg_present = {pg: False for pg in self.protecting_groups}
        
        for rxn in reactions:
            # Check all molecules in reaction (reactants and products)
            all_mols = self.get_all_molecules_from_reaction(rxn)
            
            for mol in all_mols:
                if mol is None:
                    continue
                    
                for pg_name in self.protecting_groups:
                    if pg_name in self.pg_patterns:
                        pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                        if pattern and mol.HasSubstructMatch(pattern):
                            pg_present[pg_name] = True
        
        # Check if the non-orthogonal condition is met
        if self.orthogonality == "non-orthogonal":
            # For non-orthogonal strategy, we want multiple protecting groups present
            condition = sum(pg_present.values()) >= 2
        else:
            # For orthogonal strategy, we want at most one type present
            condition = sum(pg_present.values()) <= 1
            
        return condition, len(reactions)
    
    def get_all_molecules_from_reaction(self, rxn_smiles):
        """Extract all molecules (reactants and products) from a reaction SMILES"""
        molecules = []
        try:
            if ">>" in rxn_smiles:
                reactants, products = rxn_smiles.split(">>")
                # Parse reactants
                for r_smi in reactants.split("."):
                    mol = Chem.MolFromSmiles(r_smi.strip())
                    if mol:
                        molecules.append(mol)
                # Parse products  
                for p_smi in products.split("."):
                    mol = Chem.MolFromSmiles(p_smi.strip())
                    if mol:
                        molecules.append(mol)
        except:
            pass
        return molecules
    
    def detect_protecting_group_installation(self, rxn_smiles):
        """Check if a reaction involves installation of target protecting groups"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count protecting groups in reactants vs products
            reactant_pg_count = 0
            product_pg_count = 0
            
            for pg_name in self.protecting_groups:
                if pg_name in self.pg_patterns:
                    pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                    if pattern:
                        for mol in reactant_mols:
                            if mol and mol.HasSubstructMatch(pattern):
                                reactant_pg_count += len(mol.GetSubstructMatches(pattern))
                        for mol in product_mols:
                            if mol and mol.HasSubstructMatch(pattern):
                                product_pg_count += len(mol.GetSubstructMatches(pattern))
            
            # Installation if more protecting groups in products than reactants
            return product_pg_count > reactant_pg_count
            
        except:
            return False
    
    def detect_protecting_group_removal(self, rxn_smiles):
        """Check if a reaction involves removal of target protecting groups"""
        try:
            reactants, products = rxn_smiles.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Count protecting groups in reactants vs products
            reactant_pg_count = 0
            product_pg_count = 0
            
            for pg_name in self.protecting_groups:
                if pg_name in self.pg_patterns:
                    pattern = Chem.MolFromSmarts(self.pg_patterns[pg_name])
                    if pattern:
                        for mol in reactant_mols:
                            if mol and mol.HasSubstructMatch(pattern):
                                reactant_pg_count += len(mol.GetSubstructMatches(pattern))
                        for mol in product_mols:
                            if mol and mol.HasSubstructMatch(pattern):
                                product_pg_count += len(mol.GetSubstructMatches(pattern))
            
            # Removal if fewer protecting groups in products than reactants
            return reactant_pg_count > product_pg_count
            
        except:
            return False
