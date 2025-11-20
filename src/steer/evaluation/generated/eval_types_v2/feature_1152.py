"""Generated evaluation code for: Azide reduction pathway for amine synthesis"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class AzideReductionPathway(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of azide reduction pathway for amine synthesis.
    Checks for the sequential occurrence of mesylate displacement followed by azide reduction
    to produce primary amines via stereocontrolled amine installation.
    """
    
    def __init__(self, config):
        self.require_sequence = config.get("require_sequence", True)
        self.target_functional_group = config.get("functional_group_target", "primary_amine")
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        mesylate_reactions = []
        azide_reactions = []
        primary_amine_formation = False
        
        # Identify relevant reactions and their positions
        for i, rxn in enumerate(reactions):
            if self.detect_mesylate_displacement(rxn):
                mesylate_reactions.append(i)
            if self.detect_azide_reduction(rxn):
                azide_reactions.append(i)
            if self.detect_primary_amine_formation(rxn):
                primary_amine_formation = True
        
        # Check if we have the required sequence
        has_mesylate = len(mesylate_reactions) > 0
        has_azide_reduction = len(azide_reactions) > 0
        
        if self.require_sequence:
            # Check for proper sequential order (mesylate before azide reduction)
            sequence_found = False
            for mes_idx in mesylate_reactions:
                for az_idx in azide_reactions:
                    if mes_idx < az_idx:  # Mesylate occurs before azide reduction
                        sequence_found = True
                        break
                if sequence_found:
                    break
            
            condition = sequence_found and primary_amine_formation
        else:
            condition = has_mesylate and has_azide_reduction and primary_amine_formation
        
        return condition, len(reactions)
    
    def detect_mesylate_displacement(self, rxn):
        """Detect SN2 displacement reactions involving mesylate leaving groups"""
        # Look for mesylate (methanesulfonate) pattern
        mesylate_pattern = "[CH3]S(=O)(=O)[O]"
        
        # Check if mesylate is present in reactants but not in products
        reactants_smiles = rxn.split(">>")[0]
        products_smiles = rxn.split(">>")[1]
        
        try:
            # Parse reactants
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            # Parse products
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for mesylate in reactants
            mesylate_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(mesylate_pattern))
                for mol in reactant_mols
            )
            
            # Check for azide nucleophile pattern
            azide_pattern = "[N-]=[N+]=[N-]"
            azide_nucleophile = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(azide_pattern))
                for mol in reactant_mols
            )
            
            return mesylate_in_reactants and azide_nucleophile
            
        except:
            return False
    
    def detect_azide_reduction(self, rxn):
        """Detect azide reduction to primary amine"""
        # Azide pattern
        azide_pattern = "[N-]=[N+]=[N-]"
        # Primary amine pattern  
        amine_pattern = "[CH2][NH2]"
        
        try:
            reactants_smiles = rxn.split(">>")[0]
            products_smiles = rxn.split(">>")[1]
            
            # Parse molecules
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check for azide in reactants and amine in products
            azide_in_reactants = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(azide_pattern))
                for mol in reactant_mols
            )
            
            amine_in_products = any(
                mol.HasSubstructMatch(Chem.MolFromSmarts(amine_pattern))
                for mol in product_mols
            )
            
            return azide_in_reactants and amine_in_products
            
        except:
            return False
    
    def detect_primary_amine_formation(self, rxn):
        """Detect formation of primary amine functional group"""
        primary_amine_patterns = [
            "[CH2][NH2]",  # Simple primary amine
            "[CH][NH2]",   # Primary amine on secondary carbon
            "c[NH2]",      # Aromatic primary amine
            "[NH2]"        # General primary amine
        ]
        
        try:
            products_smiles = rxn.split(">>")[1]
            
            product_mols = []
            for smi in products_smiles.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            # Check if any primary amine pattern is formed
            for pattern in primary_amine_patterns:
                if any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                       for mol in product_mols):
                    return True
            
            return False
            
        except:
            return False
