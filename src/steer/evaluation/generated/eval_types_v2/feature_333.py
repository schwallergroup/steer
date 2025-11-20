"""Generated evaluation code for: Convergent synthesis via two fragment assembly"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class ConvergentSynthesis(BaseScoring):
    """
    Evaluates convergent synthesis strategy where two advanced intermediates 
    are assembled via condensation/cyclization reaction.
    """
    
    def __init__(self, config: Dict):
        self.fragment_count = config["parameters"]["fragment_count"]
        self.assembly_step = config["parameters"]["assembly_step"]
        
        # SMARTS patterns for condensation/cyclization reactions
        self.condensation_patterns = [
            "[C:1]=[O:2].[N:3]>>[C:1][N:3]",  # Amide formation
            "[C:1]=[O:2].[O:3]>>[C:1][O:3]",  # Ester formation
            "[C:1]#[N:2].[N:3]>>[C:1]1[N:2][N:3]",  # Heterocycle formation
            "[C:1][C:2]=[O:3].[N:4]>>[C:1]1[C:2][N:4]",  # Ring closure
        ]
        
        # Patterns indicating advanced intermediates (complex structures)
        self.advanced_patterns = [
            Chem.MolFromSmarts("[R2,R3,R4]"),  # Multi-ring systems
            Chem.MolFromSmarts("c1ccccc1"),    # Aromatic rings
            Chem.MolFromSmarts("[#7,#8,#16]~[#6]~[#7,#8,#16]"),  # Heteroatom chains
            Chem.MolFromSmarts("[CH2][CH2][CH2][CH2]"),  # Long chains
        ]
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10). Earlier convergent step is better."""
        if x < 0:
            return 0  # Convergent step not found
        else:
            # Earlier convergent assembly (smaller x) gets higher score
            return max(0, 10 * (1 - x))
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction represents convergent assembly of fragments."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0]
        reactants_smiles = rxn_parts[1]
        
        # Parse reactants
        reactant_smiles_list = reactants_smiles.split(".")
        if len(reactant_smiles_list) < self.fragment_count:
            return False
            
        try:
            reactants = [Chem.MolFromSmiles(smi) for smi in reactant_smiles_list]
            reactants = [mol for mol in reactants if mol is not None]
            
            if len(reactants) < self.fragment_count:
                return False
                
            # Check if this is a condensation/cyclization reaction
            if not self._is_condensation_cyclization(mapped_rxn):
                return False
                
            # Check if reactants are advanced intermediates
            advanced_reactants = 0
            for reactant in reactants:
                if self._is_advanced_intermediate(reactant):
                    advanced_reactants += 1
                    
            # Need at least fragment_count advanced intermediates
            return advanced_reactants >= self.fragment_count
            
        except Exception:
            return False
    
    def _is_condensation_cyclization(self, mapped_rxn: str) -> bool:
        """Check if reaction matches condensation/cyclization patterns."""
        try:
            # Simple heuristics for condensation/cyclization
            rxn_parts = mapped_rxn.split(">>")
            reactants_smiles = rxn_parts[1]
            product_smiles = rxn_parts[0]
            
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            product = Chem.MolFromSmiles(product_smiles)
            
            if not all([product] + reactants):
                return False
                
            # Check for ring formation (cyclization)
            reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reactants if mol)
            product_rings = product.GetRingInfo().NumRings()
            
            if product_rings > reactant_rings:
                return True  # Ring formation detected
                
            # Check for condensation (bond formation with potential elimination)
            reactant_atoms = sum(mol.GetNumAtoms() for mol in reactants if mol)
            product_atoms = product.GetNumAtoms()
            
            # Condensation often involves elimination of small molecules
            if reactant_atoms > product_atoms and reactant_atoms - product_atoms <= 6:
                return True
                
            return False
            
        except Exception:
            return False
    
    def _is_advanced_intermediate(self, mol) -> bool:
        """Check if molecule qualifies as an advanced intermediate."""
        if not mol:
            return False
            
        try:
            # Must have reasonable complexity
            if mol.GetNumAtoms() < 6:
                return False
                
            # Check against advanced intermediate patterns
            for pattern in self.advanced_patterns:
                if pattern and mol.HasSubstructMatch(pattern):
                    return True
                    
            # Additional complexity metrics
            ring_count = mol.GetRingInfo().NumRings()
            heteroatom_count = sum(1 for atom in mol.GetAtoms() 
                                 if atom.GetAtomicNum() not in [1, 6])
            
            # Advanced if has rings or multiple heteroatoms
            return ring_count > 0 or heteroatom_count >= 2
            
        except Exception:
            return False
