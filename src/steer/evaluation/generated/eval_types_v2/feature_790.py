"""Generated evaluation code for: Multiple ester hydrolysis re-esterification cycles present"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterCycles(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of multiple ester hydrolysis 
    re-esterification cycles, typically used as protecting group swaps.
    
    Detects sequences where esters are hydrolyzed to carboxylic acids and then
    re-esterified with different alcohols, indicating protecting group manipulations.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        self.allow_multiple_cycles = config.get("allow_multiple_cycles", True)
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        cycle_count = self.count_ester_cycles(reactions)
        
        if self.allow_multiple_cycles:
            condition = cycle_count >= self.min_cycles
        else:
            condition = cycle_count < self.min_cycles
            
        return condition, len(reactions)
    
    def count_ester_cycles(self, reactions) -> int:
        """Count the number of ester hydrolysis-reesterification cycles"""
        hydrolysis_positions = []
        esterification_positions = []
        
        # Identify hydrolysis and esterification reactions
        for i, rxn in enumerate(reactions):
            if self.is_ester_hydrolysis(rxn):
                hydrolysis_positions.append(i)
            elif self.is_esterification(rxn):
                esterification_positions.append(i)
        
        # Count cycles by finding hydrolysis followed by esterification
        cycles = 0
        for hydro_pos in hydrolysis_positions:
            # Look for esterification after this hydrolysis
            for ester_pos in esterification_positions:
                if ester_pos > hydro_pos:
                    # Check if they involve similar carbon frameworks
                    if self.are_related_ester_reactions(reactions[hydro_pos], reactions[ester_pos]):
                        cycles += 1
                        break
        
        return cycles
    
    def is_ester_hydrolysis(self, rxn) -> bool:
        """Detect ester hydrolysis: R-CO-OR' + H2O -> R-COOH + R'OH"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for ester in reactants
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
            has_ester_reactant = any(mol.HasSubstructMatch(ester_pattern) for mol in reactant_mols if mol)
            
            # Check for carboxylic acid in products
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            has_acid_product = any(mol.HasSubstructMatch(acid_pattern) for mol in product_mols if mol)
            
            # Check for alcohol in products
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            has_alcohol_product = any(mol.HasSubstructMatch(alcohol_pattern) for mol in product_mols if mol)
            
            return has_ester_reactant and has_acid_product and has_alcohol_product
            
        except:
            return False
    
    def is_esterification(self, rxn) -> bool:
        """Detect esterification: R-COOH + R'OH -> R-CO-OR' + H2O"""
        try:
            reactants, products = rxn.split(">>")
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            # Check for carboxylic acid in reactants
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            has_acid_reactant = any(mol.HasSubstructMatch(acid_pattern) for mol in reactant_mols if mol)
            
            # Check for alcohol in reactants
            alcohol_pattern = Chem.MolFromSmarts("[C][OH]")
            has_alcohol_reactant = any(mol.HasSubstructMatch(alcohol_pattern) for mol in reactant_mols if mol)
            
            # Check for ester in products
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")
            has_ester_product = any(mol.HasSubstructMatch(ester_pattern) for mol in product_mols if mol)
            
            return has_acid_reactant and has_alcohol_reactant and has_ester_product
            
        except:
            return False
    
    def are_related_ester_reactions(self, hydrolysis_rxn, esterification_rxn) -> bool:
        """Check if hydrolysis and esterification reactions involve the same carbon framework"""
        try:
            # Extract the carboxylic acid from hydrolysis products
            _, hydro_products = hydrolysis_rxn.split(">>")
            hydro_mols = [Chem.MolFromSmiles(p.strip()) for p in hydro_products.split(".")]
            
            # Extract the carboxylic acid from esterification reactants  
            ester_reactants, _ = esterification_rxn.split(">>")
            ester_mols = [Chem.MolFromSmiles(r.strip()) for r in ester_reactants.split(".")]
            
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            
            # Find acids in both reactions
            hydro_acids = [mol for mol in hydro_mols if mol and mol.HasSubstructMatch(acid_pattern)]
            ester_acids = [mol for mol in ester_mols if mol and mol.HasSubstructMatch(acid_pattern)]
            
            # Check if any acid structures match (same carbon framework)
            for h_acid in hydro_acids:
                for e_acid in ester_acids:
                    if h_acid and e_acid:
                        # Simple structural comparison by canonical SMILES
                        if Chem.MolToSmiles(h_acid) == Chem.MolToSmiles(e_acid):
                            return True
            
            return False
            
        except:
            return False
