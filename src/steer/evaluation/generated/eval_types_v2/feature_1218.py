"""Generated evaluation code for: Late stage Suzuki coupling formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Evaluates whether a Suzuki-Miyaura coupling reaction occurs in the late stage of synthesis.
    
    The scoring favors routes where Suzuki coupling happens after the specified stage threshold
    (default 0.8 means the last 20% of the synthesis). Earlier occurrences receive lower scores.
    """
    
    def __init__(self, config: Dict):
        self.stage_threshold = config.get("stage_threshold", 0.8)
        
        # Suzuki coupling reactant patterns
        self.boronic_acid_pattern = Chem.MolFromSmarts("[#6]-B(O)O")
        self.boronic_ester_pattern = Chem.MolFromSmarts("[#6]-B1OC(C)(C)C(C)(C)O1")
        self.aryl_halide_pattern = Chem.MolFromSmarts("c-[Br,I,Cl]")
        self.vinyl_halide_pattern = Chem.MolFromSmarts("C=C-[Br,I,Cl]")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't occur
        
        # Score based on how late the reaction occurs
        if x >= self.stage_threshold:
            return 10  # Perfect score for late-stage coupling
        else:
            # Linearly decrease score for earlier reactions
            return 10 * (x / self.stage_threshold)

    def hit_condition(self, d):
        """Check if this reaction node represents a Suzuki coupling."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactant_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactant_smiles.split(".")]
            
            if not product_mol or not all(reactant_mols):
                return False
            
            # Check for presence of Suzuki coupling partners in reactants
            has_boron_partner = False
            has_halide_partner = False
            
            for reactant in reactant_mols:
                # Check for boronic acid or ester
                if (reactant.HasSubstructMatch(self.boronic_acid_pattern) or 
                    reactant.HasSubstructMatch(self.boronic_ester_pattern)):
                    has_boron_partner = True
                
                # Check for aryl or vinyl halide
                if (reactant.HasSubstructMatch(self.aryl_halide_pattern) or 
                    reactant.HasSubstructMatch(self.vinyl_halide_pattern)):
                    has_halide_partner = True
            
            # Suzuki coupling requires both partners
            if not (has_boron_partner and has_halide_partner):
                return False
            
            # Additional check: ensure C-C bond formation occurred
            return self._check_cc_bond_formation(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _check_cc_bond_formation(self, reactants, product):
        """Verify that a new C-C bond was formed between the coupling partners."""
        # Count aromatic and vinyl carbons in reactants vs product
        reactant_aromatic_c = sum(sum(1 for atom in mol.GetAtoms() 
                                    if atom.GetSymbol() == 'C' and atom.GetIsAromatic()) 
                                for mol in reactants)
        reactant_vinyl_c = sum(sum(1 for atom in mol.GetAtoms() 
                                 if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.HybridizationType.SP2 
                                 and not atom.GetIsAromatic()) 
                             for mol in reactants)
        
        product_aromatic_c = sum(1 for atom in product.GetAtoms() 
                               if atom.GetSymbol() == 'C' and atom.GetIsAromatic())
        product_vinyl_c = sum(1 for atom in product.GetAtoms() 
                            if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.HybridizationType.SP2 
                            and not atom.GetIsAromatic())
        
        # In Suzuki coupling, we expect similar carbon counts (allowing for some variation due to functional group changes)
        return (abs(reactant_aromatic_c - product_aromatic_c) <= 2 and 
                abs(reactant_vinyl_c - product_vinyl_c) <= 2)
