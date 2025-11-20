"""Generated evaluation code for: Convergent assembly via Suzuki cross-coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSuzukiAssembly(BaseScoring):
    """
    Evaluates convergent assembly via Suzuki-Miyaura cross-coupling.
    Checks if two advanced fragments are joined through Suzuki coupling
    at the specified timing in the synthesis route.
    """
    
    def __init__(self, config: Dict):
        self.coupling_type = config["parameters"]["coupling_type"]
        self.fragment_count = config["parameters"]["fragment_count"]
        self.timing = config["parameters"]["timing"]
        
        # Define Suzuki-Miyaura reaction patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6][B]([OH])[OH]")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6][B]1OC(C)(C)C(C)(C)O1")
        self.aryl_halide_pattern = Chem.MolFromSmarts("c[F,Cl,Br,I]")
        self.vinyl_halide_pattern = Chem.MolFromSmarts("C=C[F,Cl,Br,I]")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        # Score based on timing preference
        if self.timing == "penultimate":
            # Prefer late-stage coupling (depth close to 1.0)
            if x > 0.8:
                return 10
            elif x > 0.6:
                return 7
            elif x > 0.4:
                return 4
            else:
                return 2
        elif self.timing == "early":
            # Prefer early-stage coupling (depth close to 0.0)
            return 10 * (1 - x)
        else:
            # Any timing is acceptable
            return 8
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Suzuki-Miyaura coupling"""
        metadata = d.get("metadata", {})
        
        # Check if reaction SMILES is available
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1].split(".")
        
        # Must have exactly the specified number of main fragments
        if len(reactants_smiles) < self.fragment_count:
            return False
            
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r) for r in reactants_smiles]
            
            if not product or not all(reactants):
                return False
                
            # Check for Suzuki coupling pattern
            has_boron_source = False
            has_halide_source = False
            
            for reactant in reactants:
                # Skip small molecules (catalysts, bases, solvents)
                if reactant.GetNumHeavyAtoms() < 6:
                    continue
                    
                # Check for boronic acid/ester
                if (reactant.HasSubstructMatch(self.boronic_acid_pattern) or 
                    reactant.HasSubstructMatch(self.boronic_ester_pattern)):
                    has_boron_source = True
                    
                # Check for aryl/vinyl halide
                if (reactant.HasSubstructMatch(self.aryl_halide_pattern) or
                    reactant.HasSubstructMatch(self.vinyl_halide_pattern)):
                    has_halide_source = True
            
            # Must have both coupling partners
            if not (has_boron_source and has_halide_source):
                return False
                
            # Verify C-C bond formation occurred
            return self._verify_cc_bond_formation(reactants, product)
            
        except Exception:
            return False
    
    def _verify_cc_bond_formation(self, reactants, product) -> bool:
        """Verify that a new C-C bond was formed in the coupling"""
        # Count C-C bonds in reactants vs product
        reactant_cc_bonds = sum(self._count_cc_bonds(mol) for mol in reactants)
        product_cc_bonds = self._count_cc_bonds(product)
        
        # Should have at least one new C-C bond
        return product_cc_bonds > reactant_cc_bonds
    
    def _count_cc_bonds(self, mol) -> int:
        """Count C-C bonds in a molecule"""
        if not mol:
            return 0
            
        count = 0
        for bond in mol.GetBonds():
            if (bond.GetBeginAtom().GetAtomicNum() == 6 and 
                bond.GetEndAtom().GetAtomicNum() == 6):
                count += 1
        return count
