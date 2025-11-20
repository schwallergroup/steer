"""Generated evaluation code for: Sequential halogen exchange on biphenyl fragment"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SequentialHalogenExchange(MultiRxnCondBase):
    """
    Evaluates routes for sequential halogen exchange on biphenyl fragment.
    Checks for Suzuki coupling followed by Finkelstein reaction (Br→I exchange)
    on a biphenyl-containing molecule.
    """
    
    def __init__(self, config):
        self.target_sequence = config["parameters"]["sequence"]  # ["suzuki_coupling", "finkelstein"]
        self.fragment = config["parameters"]["fragment"]  # "biphenyl"
        
        # Define SMARTS patterns
        self.biphenyl_pattern = "c1ccccc1-c2ccccc2"
        self.suzuki_pattern = "[#5]"  # Boron for Suzuki coupling
        self.halogen_pattern = "[Br,I]"  # Bromide or Iodide for Finkelstein
        
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        # Find sequence of reactions
        suzuki_found = False
        finkelstein_found = False
        suzuki_depth = -1
        finkelstein_depth = -1
        
        for i, rxn in enumerate(reactions):
            if self.detect_suzuki_coupling(rxn):
                suzuki_found = True
                suzuki_depth = i
            elif self.detect_finkelstein_reaction(rxn) and suzuki_found:
                finkelstein_found = True
                finkelstein_depth = i
                break
        
        # Check if sequence is correct (Suzuki before Finkelstein)
        sequence_correct = (suzuki_found and finkelstein_found and 
                          suzuki_depth < finkelstein_depth)
        
        # Check if biphenyl fragment is involved
        biphenyl_involved = False
        if sequence_correct:
            for i in range(suzuki_depth, finkelstein_depth + 1):
                if self.has_biphenyl_fragment(reactions[i]):
                    biphenyl_involved = True
                    break
        
        condition_met = sequence_correct and biphenyl_involved
        depth = finkelstein_depth if condition_met else -1
        
        return condition_met, depth
    
    def detect_suzuki_coupling(self, rxn):
        """Detect Suzuki coupling by looking for boron reagent and C-C bond formation"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[1].split(".")
            products = rxn_parts[0]
            
            # Check for boron in reactants
            has_boron = False
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol and mol.HasSubstructMatch(Chem.MolFromSmarts(self.suzuki_pattern)):
                    has_boron = True
                    break
            
            # Check for biphenyl or extended aromatic system formation
            prod_mol = Chem.MolFromSmiles(products)
            if prod_mol and has_boron:
                return prod_mol.HasSubstructMatch(Chem.MolFromSmarts("c-c"))
                
        except:
            pass
        return False
    
    def detect_finkelstein_reaction(self, rxn):
        """Detect Finkelstein reaction (Br→I exchange)"""
        try:
            rxn_parts = rxn.split(">>")
            reactants = rxn_parts[1].split(".")
            products = rxn_parts[0]
            
            # Check for Br in reactants and I in products (or vice versa)
            reactant_halogens = set()
            product_halogens = set()
            
            for reactant_smiles in reactants:
                mol = Chem.MolFromSmiles(reactant_smiles)
                if mol:
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() in ['Br', 'I']:
                            reactant_halogens.add(atom.GetSymbol())
            
            prod_mol = Chem.MolFromSmiles(products)
            if prod_mol:
                for atom in prod_mol.GetAtoms():
                    if atom.GetSymbol() in ['Br', 'I']:
                        product_halogens.add(atom.GetSymbol())
            
            # Finkelstein: halogen exchange occurred
            return (len(reactant_halogens.symmetric_difference(product_halogens)) > 0 and
                    ('Br' in reactant_halogens or 'Br' in product_halogens) and
                    ('I' in reactant_halogens or 'I' in product_halogens))
            
        except:
            pass
        return False
    
    def has_biphenyl_fragment(self, rxn):
        """Check if reaction involves biphenyl fragment"""
        try:
            rxn_parts = rxn.split(">>")
            all_molecules = rxn_parts[1].split(".") + [rxn_parts[0]]
            
            biphenyl_smarts = Chem.MolFromSmarts(self.biphenyl_pattern)
            
            for mol_smiles in all_molecules:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol and mol.HasSubstructMatch(biphenyl_smarts):
                    return True
                    
        except:
            pass
        return False
