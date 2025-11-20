"""Generated evaluation code for: Sequential ester hydrolysis and re-esterification"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialEsterHydrolysisReesterification(MultiRxnCondBase):
    """
    Checks for sequential ester hydrolysis (saponification) followed by re-esterification.
    Detects routes that contain back-to-back ester hydrolysis and esterification reactions,
    typically used to convert one ester type to another (e.g., ethyl ester to methyl ester).
    """
    
    def __init__(self, config):
        self.reaction_sequence = config.get("reaction_sequence", ["saponification", "esterification"])
        self.functional_group = config.get("functional_group", "ester")
        self.sequential = config.get("sequential", True)
        
        # SMARTS patterns for ester detection
        self.ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        if len(reactions) < 2:
            return False, len(reactions)
        
        # Find hydrolysis and esterification reactions
        hydrolysis_indices = []
        esterification_indices = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_saponification(rxn):
                hydrolysis_indices.append(i)
            elif self.detect_esterification(rxn):
                esterification_indices.append(i)
        
        # Check for sequential occurrence
        if self.sequential:
            condition = self.check_sequential_reactions(hydrolysis_indices, esterification_indices)
        else:
            condition = len(hydrolysis_indices) > 0 and len(esterification_indices) > 0
        
        return condition, len(reactions)
    
    def detect_saponification(self, rxn):
        """Detect ester hydrolysis (saponification) reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Check for ester in reactants
            has_ester_reactant = any(mol.HasSubstructMatch(self.ester_pattern) for mol in reactant_mols)
            
            # Check for carboxylic acid in products
            has_acid_product = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in product_mols)
            
            # Check for water/hydroxide in reactants (typical for hydrolysis)
            has_water_or_base = any(
                Chem.MolToSmiles(mol) in ['O', '[OH-]', '[Na+].[OH-]', '[K+].[OH-]'] 
                for mol in reactant_mols
            )
            
            return has_ester_reactant and has_acid_product and has_water_or_base
            
        except:
            return False
    
    def detect_esterification(self, rxn):
        """Detect esterification reaction"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants_smiles = rxn_parts[0]
        products_smiles = rxn_parts[1]
        
        try:
            # Parse reactants and products
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            if None in reactant_mols or None in product_mols:
                return False
            
            # Check for carboxylic acid in reactants
            has_acid_reactant = any(mol.HasSubstructMatch(self.carboxylic_acid_pattern) for mol in reactant_mols)
            
            # Check for alcohol in reactants
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            has_alcohol_reactant = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactant_mols)
            
            # Check for ester in products
            has_ester_product = any(mol.HasSubstructMatch(self.ester_pattern) for mol in product_mols)
            
            return has_acid_reactant and has_alcohol_reactant and has_ester_product
            
        except:
            return False
    
    def check_sequential_reactions(self, hydrolysis_indices, esterification_indices):
        """Check if hydrolysis and esterification occur sequentially"""
        if not hydrolysis_indices or not esterification_indices:
            return False
        
        # Look for any hydrolysis followed immediately by esterification
        for h_idx in hydrolysis_indices:
            for e_idx in esterification_indices:
                if e_idx == h_idx + 1:  # Sequential reactions
                    return True
        
        return False
