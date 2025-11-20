"""Generated evaluation code for: Circular synthetic sequence via amide-nitrile cycle"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class CircularAmideNitrileSequence(MultiRxnCondBase):
    """
    Detects circular synthetic sequences involving C-COOH bond breaking
    followed by acyl chloride -> amide -> nitrile -> carboxylic acid transformations.
    """
    
    def __init__(self, config):
        self.bond_type = config.get("bond_type", "C-COOH")
        self.sequence = config.get("sequence", ["acyl_chloride", "amide", "nitrile", "carboxylic_acid"])
        self.circular = config.get("circular", True)
        
        # SMARTS patterns for functional group detection
        self.patterns = {
            "carboxylic_acid": "[C](=[O])[OH]",
            "acyl_chloride": "[C](=[O])[Cl]",
            "amide": "[C](=[O])[NH]",
            "nitrile": "[C]#[N]"
        }
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        """Check if the circular amide-nitrile sequence occurs in the route."""
        reactions = self.get_rxns(d)
        
        # Track functional group transformations
        sequence_found = self.detect_circular_sequence(reactions)
        
        condition = sequence_found if self.circular else False
        return condition, len(reactions)
    
    def detect_circular_sequence(self, reactions) -> bool:
        """Detect if the specified circular sequence occurs."""
        if len(reactions) < len(self.sequence):
            return False
            
        # Track functional group changes through reactions
        fg_sequence = []
        
        for rxn in reactions:
            rxn_parts = rxn.split(">>")
            if len(rxn_parts) != 2:
                continue
                
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Detect functional group transformation
            transformation = self.identify_transformation(reactants, products)
            if transformation:
                fg_sequence.append(transformation)
        
        # Check if sequence matches expected pattern
        return self.matches_circular_pattern(fg_sequence)
    
    def identify_transformation(self, reactants, products) -> str:
        """Identify the type of functional group transformation."""
        try:
            reactant_mols = [Chem.MolFromSmiles(s.strip()) for s in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(s.strip()) for s in products.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return None
            
            # Check what functional groups appear in products vs reactants
            reactant_fgs = set()
            product_fgs = set()
            
            for mol in reactant_mols:
                for fg_name, pattern in self.patterns.items():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        reactant_fgs.add(fg_name)
            
            for mol in product_mols:
                for fg_name, pattern in self.patterns.items():
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        product_fgs.add(fg_name)
            
            # Determine transformation type based on product formation
            if "acyl_chloride" in product_fgs and "carboxylic_acid" in reactant_fgs:
                return "acyl_chloride"
            elif "amide" in product_fgs and "acyl_chloride" in reactant_fgs:
                return "amide"
            elif "nitrile" in product_fgs and "amide" in reactant_fgs:
                return "nitrile"
            elif "carboxylic_acid" in product_fgs and "nitrile" in reactant_fgs:
                return "carboxylic_acid"
                
        except Exception:
            pass
        
        return None
    
    def matches_circular_pattern(self, fg_sequence) -> bool:
        """Check if the functional group sequence matches the expected circular pattern."""
        if len(fg_sequence) < len(self.sequence):
            return False
        
        # Look for the complete sequence in order
        target_sequence = self.sequence
        
        # Check if sequence appears consecutively
        for i in range(len(fg_sequence) - len(target_sequence) + 1):
            if fg_sequence[i:i+len(target_sequence)] == target_sequence:
                return True
        
        # For circular detection, also check if it wraps around
        if self.circular:
            extended_sequence = fg_sequence + fg_sequence[:len(target_sequence)-1]
            for i in range(len(fg_sequence)):
                if extended_sequence[i:i+len(target_sequence)] == target_sequence:
                    return True
        
        return False
