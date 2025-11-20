"""Generated evaluation code for: Sequential TMS protecting group strategy"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialTMSProtectingGroupStrategy(MultiRxnCondBase):
    """
    Evaluates whether TMS protecting groups on alkyne and alcohol substrates
    are removed in separate sequential steps rather than simultaneously.
    """
    
    def __init__(self, config):
        self.protecting_group = config.get("protecting_group", "TMS")
        self.strategy = config.get("strategy", "sequential_deprotection")
        self.substrates = config.get("substrates", ["alkyne", "alcohol"])
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find reactions that remove TMS from alkyne or alcohol
        tms_alkyne_deprotections = []
        tms_alcohol_deprotections = []
        
        for i, rxn in enumerate(reactions):
            if self.detect_tms_alkyne_deprotection(rxn):
                tms_alkyne_deprotections.append(i)
            if self.detect_tms_alcohol_deprotection(rxn):
                tms_alcohol_deprotections.append(i)
        
        # Check if both types of deprotections occur
        has_alkyne_deprotection = len(tms_alkyne_deprotections) > 0
        has_alcohol_deprotection = len(tms_alcohol_deprotections) > 0
        
        if not (has_alkyne_deprotection and has_alcohol_deprotection):
            # Strategy not applicable - missing one or both deprotection types
            return False, len(reactions)
        
        # Check if deprotections are sequential (not in the same reaction)
        sequential = not any(i in tms_alcohol_deprotections for i in tms_alkyne_deprotections)
        
        return sequential, len(reactions)
    
    def detect_tms_alkyne_deprotection(self, rxn):
        """Detect removal of TMS group from terminal alkyne."""
        # TMS-protected alkyne pattern: C#C[Si](C)(C)C
        tms_alkyne_pattern = "C#C[Si](C)(C)C"
        # Free terminal alkyne pattern: C#C
        free_alkyne_pattern = "C#C"
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if reactants contain TMS-protected alkyne
            has_tms_alkyne_reactant = False
            for reactant_smiles in reactants:
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant and reactant.HasSubstructMatch(Chem.MolFromSmarts(tms_alkyne_pattern)):
                    has_tms_alkyne_reactant = True
                    break
            
            # Check if products contain free terminal alkyne
            has_free_alkyne_product = False
            for product_smiles in products:
                product = Chem.MolFromSmiles(product_smiles)
                if product and product.HasSubstructMatch(Chem.MolFromSmarts(free_alkyne_pattern)):
                    # Ensure it's actually terminal (not internal)
                    if self.has_terminal_alkyne(product):
                        has_free_alkyne_product = True
                        break
            
            return has_tms_alkyne_reactant and has_free_alkyne_product
            
        except:
            return False
    
    def detect_tms_alcohol_deprotection(self, rxn):
        """Detect removal of TMS group from alcohol."""
        # TMS-protected alcohol pattern: O[Si](C)(C)C
        tms_alcohol_pattern = "O[Si](C)(C)C"
        # Free alcohol pattern: [OH]
        free_alcohol_pattern = "[OH]"
        
        try:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants = rxn_parts[0].split(".")
            products = rxn_parts[1].split(".")
            
            # Check if reactants contain TMS-protected alcohol
            has_tms_alcohol_reactant = False
            for reactant_smiles in reactants:
                reactant = Chem.MolFromSmiles(reactant_smiles)
                if reactant and reactant.HasSubstructMatch(Chem.MolFromSmarts(tms_alcohol_pattern)):
                    has_tms_alcohol_reactant = True
                    break
            
            # Check if products contain free alcohol
            has_free_alcohol_product = False
            for product_smiles in products:
                product = Chem.MolFromSmiles(product_smiles)
                if product and product.HasSubstructMatch(Chem.MolFromSmarts(free_alcohol_pattern)):
                    has_free_alcohol_product = True
                    break
            
            return has_tms_alcohol_reactant and has_free_alcohol_product
            
        except:
            return False
    
    def has_terminal_alkyne(self, mol):
        """Check if molecule has a terminal alkyne (C#C with one carbon having only one heavy atom neighbor)."""
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                
                # Check if either carbon in the triple bond has only one heavy atom neighbor
                if (atom1.GetSymbol() == 'C' and len([n for n in atom1.GetNeighbors() if n.GetAtomicNum() > 1]) == 1) or \
                   (atom2.GetSymbol() == 'C' and len([n for n in atom2.GetNeighbors() if n.GetAtomicNum() > 1]) == 1):
                    return True
        return False
