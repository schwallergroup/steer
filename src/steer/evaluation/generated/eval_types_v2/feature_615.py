"""Generated evaluation code for: Multiple ester interconversion cycles"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class MultipleEsterInterconversionCycles(MultiRxnCondBase):
    """
    Detects routes with multiple cycles of ester formation and hydrolysis reactions.
    A cycle is defined as converting between different ester forms or between ester and carboxylic acid.
    """
    
    def __init__(self, config):
        self.min_cycles = config.get("min_cycles", 2)
        self.ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C]")  # Ester functional group
        self.carboxylic_acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")  # Carboxylic acid
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        ester_interconversion_count = self.count_ester_cycles(reactions)
        
        condition = ester_interconversion_count >= self.min_cycles
        return condition, len(reactions)
    
    def count_ester_cycles(self, reactions) -> int:
        """Count the number of ester interconversion cycles in the reaction sequence."""
        ester_reactions = []
        
        for rxn in reactions:
            if self.is_ester_interconversion(rxn):
                ester_reactions.append(rxn)
        
        # Count cycles: consecutive ester interconversions
        if len(ester_reactions) < 2:
            return 0
            
        # A cycle requires at least 2 ester interconversion reactions
        # Count pairs of consecutive ester interconversions as cycles
        cycles = 0
        consecutive_count = 1
        
        for i in range(1, len(ester_reactions)):
            # Check if reactions are related (involve similar carbon frameworks)
            if self.are_related_ester_reactions(ester_reactions[i-1], ester_reactions[i]):
                consecutive_count += 1
            else:
                if consecutive_count >= 2:
                    cycles += consecutive_count - 1
                consecutive_count = 1
        
        # Check the final sequence
        if consecutive_count >= 2:
            cycles += consecutive_count - 1
            
        return cycles
    
    def is_ester_interconversion(self, rxn) -> bool:
        """Check if a reaction involves ester formation or hydrolysis."""
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi.strip()) for smi in products_smiles.split(".")]
            
            reactant_mols = [mol for mol in reactant_mols if mol is not None]
            product_mols = [mol for mol in product_mols if mol is not None]
            
            # Count esters and carboxylic acids in reactants and products
            reactant_esters = sum(1 for mol in reactant_mols if mol.HasSubstructMatch(self.ester_pattern))
            reactant_acids = sum(1 for mol in reactant_mols if mol.HasSubstructMatch(self.carboxylic_acid_pattern))
            
            product_esters = sum(1 for mol in product_mols if mol.HasSubstructMatch(self.ester_pattern))
            product_acids = sum(1 for mol in product_mols if mol.HasSubstructMatch(self.carboxylic_acid_pattern))
            
            # Esterification: acid -> ester (increase in esters, decrease in acids)
            # Hydrolysis: ester -> acid (decrease in esters, increase in acids)
            # Transesterification: ester -> different ester (ester count same but different R groups)
            
            esterification = (product_esters > reactant_esters) and (reactant_acids > product_acids)
            hydrolysis = (reactant_esters > product_esters) and (product_acids > reactant_acids)
            transesterification = (reactant_esters > 0) and (product_esters > 0) and self.has_different_ester_groups(reactant_mols, product_mols)
            
            return esterification or hydrolysis or transesterification
            
        except:
            return False
    
    def has_different_ester_groups(self, reactant_mols, product_mols) -> bool:
        """Check if ester groups have different alkyl substituents between reactants and products."""
        try:
            # Simple heuristic: different molecular formulas suggest different ester groups
            reactant_formulas = set()
            product_formulas = set()
            
            for mol in reactant_mols:
                if mol.HasSubstructMatch(self.ester_pattern):
                    reactant_formulas.add(Chem.rdMolDescriptors.CalcMolFormula(mol))
                    
            for mol in product_mols:
                if mol.HasSubstructMatch(self.ester_pattern):
                    product_formulas.add(Chem.rdMolDescriptors.CalcMolFormula(mol))
            
            return len(reactant_formulas.intersection(product_formulas)) == 0
            
        except:
            return False
    
    def are_related_ester_reactions(self, rxn1, rxn2) -> bool:
        """Check if two ester reactions are related (part of the same interconversion cycle)."""
        try:
            # Extract the carbon skeleton from both reactions to see if they're related
            rxn1_mols = self.get_all_mols_from_reaction(rxn1)
            rxn2_mols = self.get_all_mols_from_reaction(rxn2)
            
            # Simple heuristic: if reactions share similar molecular weights or carbon counts
            rxn1_carbon_counts = set()
            rxn2_carbon_counts = set()
            
            for mol in rxn1_mols:
                if mol and mol.HasSubstructMatch(self.ester_pattern):
                    rxn1_carbon_counts.add(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'))
                    
            for mol in rxn2_mols:
                if mol and mol.HasSubstructMatch(self.ester_pattern):
                    rxn2_carbon_counts.add(sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C'))
            
            return len(rxn1_carbon_counts.intersection(rxn2_carbon_counts)) > 0
            
        except:
            return False
    
    def get_all_mols_from_reaction(self, rxn):
        """Extract all molecules from a reaction SMILES."""
        try:
            rxn_parts = rxn.split(">>")
            all_smiles = (rxn_parts[0] + "." + rxn_parts[1]).split(".")
            return [Chem.MolFromSmiles(smi.strip()) for smi in all_smiles if smi.strip()]
        except:
            return []
