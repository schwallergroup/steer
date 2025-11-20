"""Generated evaluation code for: Benzyl ether deprotection followed by methylation sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class BenzylEtherDeprotectionMethylation(MultiRxnCondBase):
    """
    Evaluates routes for benzyl ether deprotection followed by methylation sequence.
    Checks if the route contains deprotection of benzyl ether to phenol followed by
    methylation to form methyl ether in the correct sequence.
    """
    
    def __init__(self, config):
        self.strategy_type = config.get("strategy_type", "sequential_swap")
        self.sequence = config.get("sequence", "deprotect_then_protect")
        
        # SMARTS patterns for detecting transformations
        self.benzyl_ether_pattern = "[OH1][c:1]1[c:2][c:3][c:4][c:5][c:6]1"  # Phenol
        self.methyl_ether_pattern = "[CH3][OH0][c:1]1[c:2][c:3][c:4][c:5][c:6]1"  # Methyl ether
        self.benzyl_group_pattern = "[CH2][c:1]1[c:2][c:3][c:4][c:5][c:6]1"  # Benzyl group

    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if the route contains the benzyl ether deprotection -> methylation sequence.
        Returns (condition_met, total_reactions).
        """
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Find benzyl deprotection and methylation reactions
        deprotection_indices = []
        methylation_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_benzyl_deprotection(rxn):
                deprotection_indices.append(i)
            elif self.detect_methylation(rxn):
                methylation_indices.append(i)
        
        # Check if we have the correct sequence
        sequence_found = False
        if self.sequence == "deprotect_then_protect":
            # Look for deprotection followed by methylation
            for dep_idx in deprotection_indices:
                for meth_idx in methylation_indices:
                    # In synthesis routes, later reactions have higher indices
                    if meth_idx > dep_idx:
                        # Verify they act on the same or related phenolic position
                        if self.verify_sequential_transformation(reactions[dep_idx], reactions[meth_idx]):
                            sequence_found = True
                            break
                if sequence_found:
                    break
        
        return sequence_found, len(reactions)

    def detect_benzyl_deprotection(self, rxn):
        """Detect benzyl ether deprotection (benzyl ether -> phenol)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Check for benzyl group in reactants and phenol in products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Look for benzyl ether in reactants
            has_benzyl_ether = False
            for mol in reactant_mols:
                if mol and self.has_benzyl_ether_substructure(mol):
                    has_benzyl_ether = True
                    break
            
            # Look for phenol in products
            has_phenol = False
            for mol in product_mols:
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)):
                    has_phenol = True
                    break
            
            return has_benzyl_ether and has_phenol
            
        except:
            return False

    def detect_methylation(self, rxn):
        """Detect methylation of phenol to methyl ether"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Look for phenol in reactants
            has_phenol = False
            for mol in reactant_mols:
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.benzyl_ether_pattern)):
                    has_phenol = True
                    break
            
            # Look for methyl ether in products
            has_methyl_ether = False
            for mol in product_mols:
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.methyl_ether_pattern)):
                    has_methyl_ether = True
                    break
            
            return has_phenol and has_methyl_ether
            
        except:
            return False

    def has_benzyl_ether_substructure(self, mol):
        """Check if molecule contains benzyl ether group"""
        benzyl_ether_smarts = "[CH2][c:1]1[c:2][c:3][c:4][c:5][c:6]1.[OH0][c:7]"
        try:
            pattern = Chem.MolFromSmarts(benzyl_ether_smarts)
            return mol.HasSubstructMatch(pattern)
        except:
            # Fallback: look for benzyl and ether separately
            benzyl_pattern = Chem.MolFromSmarts(self.benzyl_group_pattern)
            ether_pattern = Chem.MolFromSmarts("[OH0][c]")
            return mol.HasSubstructMatch(benzyl_pattern) and mol.HasSubstructMatch(ether_pattern)

    def verify_sequential_transformation(self, deprotection_rxn, methylation_rxn):
        """
        Verify that deprotection and methylation act on the same phenolic position.
        This is a simplified check based on reaction connectivity.
        """
        try:
            # Extract the phenol intermediate from deprotection products
            dep_products = deprotection_rxn.split(">>")[1]
            dep_product_mols = [Chem.MolFromSmiles(p.strip()) for p in dep_products.split(".")]
            
            # Extract the phenol reactant from methylation
            meth_reactants = methylation_rxn.split(">>")[0]
            meth_reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in meth_reactants.split(".")]
            
            # Look for phenol in both
            dep_phenol = None
            meth_phenol = None
            
            phenol_pattern = Chem.MolFromSmarts(self.benzyl_ether_pattern)
            
            for mol in dep_product_mols:
                if mol and mol.HasSubstructMatch(phenol_pattern):
                    dep_phenol = mol
                    break
                    
            for mol in meth_reactant_mols:
                if mol and mol.HasSubstructMatch(phenol_pattern):
                    meth_phenol = mol
                    break
            
            # Simple structural similarity check
            if dep_phenol and meth_phenol:
                return Chem.MolToSmiles(dep_phenol) == Chem.MolToSmiles(meth_phenol)
            
            return Tru
