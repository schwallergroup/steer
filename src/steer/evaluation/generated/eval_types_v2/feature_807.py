"""Generated evaluation code for: Dual cyclopropanation approach"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DualCyclopropanation(MultiRxnCondBase):
    """
    Evaluates synthesis routes for dual cyclopropanation approach.
    Checks if exactly 2 cyclopropanation reactions occur using specified methods
    (sulfur ylide or Simmons-Smith reactions).
    """
    
    def __init__(self, config):
        self.target_count = config["parameters"]["count"]
        self.allowed_methods = config["parameters"]["methods"]
        self.allow_sulfur_ylide = "sulfur_ylide" in self.allowed_methods
        self.allow_simmons_smith = "simmons_smith" in self.allowed_methods
    
    def condition_depth(self, d) -> Tuple[bool, int]:
        reactions = self.get_rxns(d)
        
        cyclopropanation_count = 0
        valid_method_used = False
        
        for rxn in reactions:
            if self.detect_cyclopropanation(rxn):
                cyclopropanation_count += 1
                if self.is_valid_cyclopropanation_method(rxn):
                    valid_method_used = True
        
        # Condition met if we have exactly the target count and used valid methods
        condition = (cyclopropanation_count == self.target_count and 
                    valid_method_used and 
                    cyclopropanation_count > 0)
        
        return condition, len(reactions)
    
    def detect_cyclopropanation(self, rxn):
        """Detect if reaction forms a cyclopropyl ring"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        products = rxn_parts[1]
        
        try:
            # Count cyclopropyl rings in reactants vs products
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            product_mols = [Chem.MolFromSmiles(p.strip()) for p in products.split(".")]
            
            reactant_cyclopropyl_count = sum(self.count_cyclopropyl_rings(mol) 
                                           for mol in reactant_mols if mol)
            product_cyclopropyl_count = sum(self.count_cyclopropyl_rings(mol) 
                                          for mol in product_mols if mol)
            
            return product_cyclopropyl_count > reactant_cyclopropyl_count
            
        except:
            return False
    
    def count_cyclopropyl_rings(self, mol):
        """Count number of cyclopropyl rings in molecule"""
        if not mol:
            return 0
            
        cyclopropyl_pattern = Chem.MolFromSmarts("[C;R1]1[C;R1][C;R1]1")
        if not cyclopropyl_pattern:
            return 0
            
        matches = mol.GetSubstructMatches(cyclopropyl_pattern)
        return len(matches)
    
    def is_valid_cyclopropanation_method(self, rxn):
        """Check if cyclopropanation uses allowed methods"""
        if self.allow_sulfur_ylide and self.detect_sulfur_ylide_cyclopropanation(rxn):
            return True
        if self.allow_simmons_smith and self.detect_simmons_smith_cyclopropanation(rxn):
            return True
        return False
    
    def detect_sulfur_ylide_cyclopropanation(self, rxn):
        """Detect sulfur ylide cyclopropanation (presence of sulfur ylide reagent)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            # Look for sulfur ylide pattern: S with positive charge adjacent to C with negative charge
            sulfur_ylide_patterns = [
                Chem.MolFromSmarts("[S+]-[C-]"),  # Simple ylide
                Chem.MolFromSmarts("[S+]([C-])([C,c])([C,c])"),  # Substituted ylide
                Chem.MolFromSmarts("S(=C)([C,c])([C,c])")  # Alternative representation
            ]
            
            for mol in reactant_mols:
                if mol:
                    for pattern in sulfur_ylide_patterns:
                        if pattern and mol.HasSubstructMatch(pattern):
                            return True
            
            return False
            
        except:
            return False
    
    def detect_simmons_smith_cyclopropanation(self, rxn):
        """Detect Simmons-Smith cyclopropanation (zinc carbenoid reagents)"""
        rxn_parts = rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        reactants = rxn_parts[0]
        
        try:
            reactant_mols = [Chem.MolFromSmiles(r.strip()) for r in reactants.split(".")]
            
            # Look for common Simmons-Smith reagents
            simmons_smith_patterns = [
                Chem.MolFromSmarts("[Zn]"),  # Zinc
                Chem.MolFromSmarts("BrC[Br]"),  # Dibromomethane
                Chem.MolFromSmarts("IC[I]"),  # Diiodomethane
                Chem.MolFromSmarts("[Zn][CH2][Zn]")  # Zinc carbenoid
            ]
            
            for mol in reactant_mols:
                if mol:
                    for pattern in simmons_smith_patterns:
                        if pattern and mol.HasSubstructMatch(pattern):
                            return True
            
            return False
            
        except:
            return False
