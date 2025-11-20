"""Generated evaluation code for: Strategic carboxylic acid protection throughout metal coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CarboxylicAcidProtectionStrategy(MultiRxnCondBase):
    """
    Evaluates whether carboxylic acids are properly protected as methyl esters
    during metal coupling reactions (e.g., Suzuki coupling) in the synthesis route.
    """
    
    def __init__(self, config):
        self.require_protection = config.get("require_protection", True)
        self.carboxylic_acid_pattern = "[CX3](=O)[OX2H1]"  # Carboxylic acid SMARTS
        self.methyl_ester_pattern = "[CX3](=O)[OX2][CH3]"  # Methyl ester SMARTS
        self.metal_coupling_patterns = [
            "[c,C]-[Pd]",  # Palladium coupling
            "[c,C]-B([OH])[OH]",  # Boronic acid (Suzuki)
            "[c,C]-B1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
            "[c,C][Sn]([C,c])([C,c])[C,c]",  # Stannane (Stille)
            "[c,C]-[Zn]",  # Organozinc (Negishi)
        ]
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """
        Check if carboxylic acids are protected during metal coupling reactions.
        Returns (condition_met, total_reactions).
        """
        reactions = self.get_rxns(d)
        total_reactions = len(reactions)
        
        if total_reactions == 0:
            return True, 0
        
        violations = 0
        
        for rxn in reactions:
            if self.is_metal_coupling(rxn):
                # Check if unprotected carboxylic acids are present
                has_unprotected_acid = self.has_unprotected_carboxylic_acid(rxn)
                
                if self.require_protection and has_unprotected_acid:
                    violations += 1
                elif not self.require_protection and not has_unprotected_acid:
                    violations += 1
        
        condition_met = violations == 0
        return condition_met, total_reactions
    
    def is_metal_coupling(self, rxn):
        """Check if the reaction involves metal coupling chemistry."""
        reactants_smiles = rxn.split(">>")[0]
        products_smiles = rxn.split(">>")[1]
        
        # Parse all molecules in reactants and products
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
        
        # Check for metal coupling patterns in reactants
        for mol in reactant_mols:
            for pattern in self.metal_coupling_patterns:
                if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                    return True
        
        # Additional check: C-C bond formation between aromatic/sp2 carbons
        # This is characteristic of metal coupling reactions
        return self.detect_cc_coupling(reactant_mols, product_mols)
    
    def detect_cc_coupling(self, reactant_mols, product_mols):
        """Detect C-C bond formation typical of metal coupling."""
        # Simple heuristic: check if aromatic carbon count increases
        # in a way consistent with coupling
        reactant_aromatic_c = sum(
            len([a for a in mol.GetAtoms() if a.GetIsAromatic() and a.GetSymbol() == 'C'])
            for mol in reactant_mols
        )
        
        product_aromatic_c = sum(
            len([a for a in mol.GetAtoms() if a.GetIsAromatic() and a.GetSymbol() == 'C'])
            for mol in product_mols
        )
        
        # If aromatic carbon count is conserved, likely coupling occurred
        return abs(reactant_aromatic_c - product_aromatic_c) <= 2
    
    def has_unprotected_carboxylic_acid(self, rxn):
        """Check if reaction involves molecules with unprotected carboxylic acids."""
        all_smiles = rxn.replace(">>", ".")
        
        for smi in all_smiles.split("."):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # Check for carboxylic acid
                if mol.HasSubstructMatch(Chem.MolFromSmarts(self.carboxylic_acid_pattern)):
                    # Check if there's also a methyl ester (partial protection)
                    if not mol.HasSubstructMatch(Chem.MolFromSmarts(self.methyl_ester_pattern)):
                        return True
                    # Even if methyl ester present, unprotected acid is still problematic
                    return True
        
        return False
