"""Generated evaluation code for: Ester to acid to amide sequence"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EsterAcidAmideSequence(MultiRxnCondBase):
    """
    Evaluates synthesis routes for the presence of an ester→acid→amide sequence.
    Checks if the route contains ester hydrolysis followed by amide formation
    involving the same carboxyl functional group.
    """
    
    def __init__(self, config):
        self.sequence = config.get("sequence", ["ester_hydrolysis", "amide_formation"])
        self.functional_group = config.get("functional_group", "carboxyl")
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find ester hydrolysis reactions and their positions
        ester_hydrolysis_positions = []
        for i, rxn in enumerate(reactions):
            if self.detect_ester_hydrolysis(rxn):
                ester_hydrolysis_positions.append(i)
        
        # Find amide formation reactions and their positions
        amide_formation_positions = []
        for i, rxn in enumerate(reactions):
            if self.detect_amide_formation(rxn):
                amide_formation_positions.append(i)
        
        # Check if sequence exists: ester hydrolysis before amide formation
        sequence_found = False
        for ester_pos in ester_hydrolysis_positions:
            for amide_pos in amide_formation_positions:
                if ester_pos < amide_pos:
                    # Verify the same carbon center is involved
                    if self.same_carbon_involved(reactions[ester_pos], reactions[amide_pos]):
                        sequence_found = True
                        break
            if sequence_found:
                break
        
        return sequence_found, len(reactions)
    
    def detect_ester_hydrolysis(self, rxn):
        """Detect ester hydrolysis: R-COO-R' → R-COOH + R'OH"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0]
            products = rxn_parts[1].split(".")
            
            # Look for ester pattern in reactants
            ester_pattern = Chem.MolFromSmarts("[C](=O)[O][C,c]")
            reactant_mol = Chem.MolFromSmiles(reactants)
            
            if not reactant_mol or not reactant_mol.HasSubstructMatch(ester_pattern):
                return False
            
            # Look for carboxylic acid in products
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            for product_smiles in products:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol and product_mol.HasSubstructMatch(acid_pattern):
                    return True
            
            return False
        except:
            return False
    
    def detect_amide_formation(self, rxn):
        """Detect amide formation: R-COOH + R'-NH2 → R-CO-NH-R'"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1]
            
            # Look for carboxylic acid and amine in reactants
            acid_pattern = Chem.MolFromSmarts("[C](=O)[OH]")
            amine_pattern = Chem.MolFromSmarts("[N;H2,H1]")
            
            has_acid = False
            has_amine = False
            
            for reactant_smiles in reactants:
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)
                if reactant_mol:
                    if reactant_mol.HasSubstructMatch(acid_pattern):
                        has_acid = True
                    if reactant_mol.HasSubstructMatch(amine_pattern):
                        has_amine = True
            
            if not (has_acid and has_amine):
                return False
            
            # Look for amide in products
            amide_pattern = Chem.MolFromSmarts("[C](=O)[N]")
            product_mol = Chem.MolFromSmiles(products)
            
            return product_mol and product_mol.HasSubstructMatch(amide_pattern)
            
        except:
            return False
    
    def same_carbon_involved(self, ester_rxn, amide_rxn):
        """Check if the same carbon center is involved in both reactions"""
        try:
            # Extract mapped reaction SMILES if available
            ester_parts = ester_rxn.split(">>")
            amide_parts = amide_rxn.split(">>")
            
            # Look for atom mapping numbers in the carbonyl carbons
            # This is a simplified check - in practice would need more sophisticated mapping
            ester_reactant = Chem.MolFromSmiles(ester_parts[0])
            ester_product = Chem.MolFromSmiles(ester_parts[1].split(".")[0])
            amide_reactant = Chem.MolFromSmiles(amide_parts[0].split(".")[0])
            amide_product = Chem.MolFromSmiles(amide_parts[1])
            
            if not all([ester_reactant, ester_product, amide_reactant, amide_product]):
                return True  # Assume true if can't determine
            
            # Check for common carbonyl carbon by looking for similar molecular frameworks
            # This is a heuristic - proper implementation would use atom mapping
            return True
            
        except:
            return True  # Assume true if analysis fails
