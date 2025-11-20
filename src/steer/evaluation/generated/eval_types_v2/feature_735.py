"""Generated evaluation code for: Williamson ether synthesis for fragment coupling"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class WilliamsonEtherSynthesis(BaseScoring):
    """
    Evaluates synthesis routes for Williamson ether synthesis reactions.
    Detects C-O bond formation through nucleophilic substitution between
    an alkoxide/phenoxide and an alkyl halide or similar electrophile.
    Rewards early-stage fragment coupling via ether formation.
    """
    
    def __init__(self, config: Dict):
        self.stage_preference = config["parameters"].get("stage", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Williamson ether synthesis doesn't occur
        
        if self.stage_preference == "early":
            return 1 - x  # Early-stage is better (lower depth fraction)
        elif self.stage_preference == "late":
            return x  # Late-stage is better (higher depth fraction)
        else:
            return 1.0  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """
        Detects Williamson ether synthesis by looking for:
        1. Formation of C-O-C ether linkage
        2. Alkyl halide or tosylate electrophile pattern
        3. Nucleophilic substitution mechanism indicators
        """
        try:
            rxn_smiles = d["metadata"]["mapped_reaction_smiles"]
            reactants_smiles, products_smiles = rxn_smiles.split(">>")
            
            # Parse molecules
            reactants = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactants) or not all(products):
                return False
            
            # Check for ether formation (C-O-C bond creation)
            if not self._has_ether_formation(reactants, products):
                return False
            
            # Check for characteristic electrophilic patterns
            has_electrophile = any(self._has_electrophile_pattern(mol) for mol in reactants)
            
            # Check for nucleophilic oxygen patterns (alkoxide, phenoxide, hydroxyl)
            has_nucleophile = any(self._has_oxygen_nucleophile(mol) for mol in reactants)
            
            return has_electrophile and has_nucleophile
            
        except Exception:
            return False
    
    def _has_ether_formation(self, reactants, products):
        """Check if C-O-C ether bonds are formed in the reaction"""
        # Count ether linkages in reactants vs products
        reactant_ethers = sum(self._count_ether_bonds(mol) for mol in reactants)
        product_ethers = sum(self._count_ether_bonds(mol) for mol in products)
        
        return product_ethers > reactant_ethers
    
    def _count_ether_bonds(self, mol):
        """Count C-O-C ether linkages in a molecule"""
        if not mol:
            return 0
            
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'O' and atom.GetDegree() == 2:
                # Check if oxygen is bonded to two carbons
                neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
                if len(neighbors) == 2 and all(n.GetSymbol() == 'C' for n in neighbors):
                    count += 1
        return count
    
    def _has_electrophile_pattern(self, mol):
        """Detect alkyl halides, tosylates, and other good leaving groups"""
        if not mol:
            return False
        
        # Alkyl halides (C-X where X = Cl, Br, I)
        alkyl_halide_patterns = [
            "[CH3][Cl,Br,I]",  # Primary
            "[CH2][Cl,Br,I]",  # Primary
            "[CH1][Cl,Br,I]",  # Secondary
            "C[Cl,Br,I]"       # General carbon-halogen
        ]
        
        # Tosylates and mesylates
        sulfonate_patterns = [
            "COS(=O)(=O)c1ccc(C)cc1",  # Tosylate
            "COS(=O)(=O)C"             # Mesylate
        ]
        
        all_patterns = alkyl_halide_patterns + sulfonate_patterns
        
        for pattern in all_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        
        return False
    
    def _has_oxygen_nucleophile(self, mol):
        """Detect oxygen nucleophiles: alcohols, phenols, alkoxides"""
        if not mol:
            return False
        
        nucleophile_patterns = [
            "[OH]",           # Hydroxyl group
            "[O-]",           # Alkoxide anion
            "c[OH]",          # Phenol
            "c[O-]",          # Phenoxide
            "[CH3][OH]",      # Methanol
            "[CH2][OH]",      # Primary alcohol
            "[CH1][OH]"       # Secondary alcohol
        ]
        
        for pattern in nucleophile_patterns:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                return True
        
        return False
