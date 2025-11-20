"""Generated evaluation code for: Sonogashira coupling for alkyne installation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class SonogashiraCoupling(BaseScoring):
    """
    Evaluates synthesis routes for the presence of Sonogashira coupling reactions.
    
    Sonogashira coupling involves the palladium-catalyzed cross-coupling of 
    aryl/vinyl halides with terminal alkynes to form C-C bonds with alkyne functionality.
    """
    
    def __init__(self, config: Dict):
        self.reaction_smarts = config["parameters"]["reaction_smarts"]
        self.reaction_name = config["parameters"]["reaction_name"]
        
        # Parse the reaction SMARTS pattern
        self.rxn_pattern = AllChem.ReactionFromSmarts(self.reaction_smarts)
        
        # Alternative patterns for Sonogashira reactions
        self.alt_patterns = [
            "[c:1][Br].[C:2]#[C:3]>>[c:1][C:2]#[C:3]",  # Aryl bromide variant
            "[c:1][Cl].[C:2]#[C:3]>>[c:1][C:2]#[C:3]",  # Aryl chloride variant
            "[C:1]=[C:2][I].[C:3]#[C:4]>>[C:1]=[C:2][C:3]#[C:4]"  # Vinyl iodide variant
        ]
        self.alt_rxn_patterns = [AllChem.ReactionFromSmarts(smarts) for smarts in self.alt_patterns]
    
    def route_scoring(self, x) -> float:
        """Convert depth fraction to score (0-10 scale)"""
        if x < 0:
            return 0  # Reaction not found
        else:
            # Earlier use of Sonogashira is generally better for synthetic planning
            return 1 - x
    
    def hit_condition(self, d) -> bool:
        """Check if a reaction node represents a Sonogashira coupling"""
        try:
            mapped_rxn_smiles = d.get("metadata", {}).get("mapped_reaction_smiles")
            if not mapped_rxn_smiles:
                return False
            
            # Parse reaction SMILES
            rxn_parts = mapped_rxn_smiles.split(">>")
            if len(rxn_parts) != 2:
                return False
            
            reactants_smiles = rxn_parts[0]
            products_smiles = rxn_parts[1]
            
            # Create reaction object
            test_rxn = AllChem.ReactionFromSmarts(f"{reactants_smiles}>>{products_smiles}")
            if test_rxn is None:
                return False
            
            # Check against main pattern
            if self._matches_reaction_pattern(test_rxn, self.rxn_pattern):
                return True
            
            # Check against alternative patterns
            for alt_pattern in self.alt_rxn_patterns:
                if self._matches_reaction_pattern(test_rxn, alt_pattern):
                    return True
            
            # Additional heuristic check for Sonogashira characteristics
            return self._heuristic_sonogashira_check(reactants_smiles, products_smiles)
            
        except Exception:
            return False
    
    def _matches_reaction_pattern(self, test_rxn, pattern_rxn):
        """Check if test reaction matches the pattern reaction"""
        try:
            # Simple approach: check if the reaction has the right atom types and connectivity
            if test_rxn.GetNumReactantTemplates() != pattern_rxn.GetNumReactantTemplates():
                return False
            if test_rxn.GetNumProductTemplates() != pattern_rxn.GetNumProductTemplates():
                return False
            
            # For Sonogashira, we expect: aryl halide + terminal alkyne -> aryl alkyne
            return self._check_sonogashira_transformation(test_rxn)
            
        except Exception:
            return False
    
    def _check_sonogashira_transformation(self, rxn):
        """Check for characteristic Sonogashira transformation patterns"""
        try:
            # Look for aryl halide in reactants and alkyne in both reactants and products
            reactant_templates = [rxn.GetReactantTemplate(i) for i in range(rxn.GetNumReactantTemplates())]
            product_templates = [rxn.GetProductTemplate(i) for i in range(rxn.GetNumProductTemplates())]
            
            # Check for aryl halide pattern in reactants
            aryl_halide_pattern = Chem.MolFromSmarts("[c][I,Br,Cl]")
            has_aryl_halide = any(mol.HasSubstructMatch(aryl_halide_pattern) for mol in reactant_templates if mol)
            
            # Check for terminal alkyne in reactants
            terminal_alkyne_pattern = Chem.MolFromSmarts("[C]#[CH]")
            has_terminal_alkyne = any(mol.HasSubstructMatch(terminal_alkyne_pattern) for mol in reactant_templates if mol)
            
            # Check for internal alkyne in products (C#C-C pattern)
            internal_alkyne_pattern = Chem.MolFromSmarts("[c][C]#[C]")
            has_internal_alkyne = any(mol.HasSubstructMatch(internal_alkyne_pattern) for mol in product_templates if mol)
            
            return has_aryl_halide and has_terminal_alkyne and has_internal_alkyne
            
        except Exception:
            return False
    
    def _heuristic_sonogashira_check(self, reactants_smiles, products_smiles):
        """Heuristic check based on SMILES analysis"""
        try:
            # Parse reactants
            reactant_mols = [Chem.MolFromSmiles(smi) for smi in reactants_smiles.split(".")]
            product_mols = [Chem.MolFromSmiles(smi) for smi in products_smiles.split(".")]
            
            if not all(reactant_mols) or not all(product_mols):
                return False
            
            # Check for halogen loss (I, Br, Cl should be in reactants but not products)
            reactant_atoms = set()
            product_atoms = set()
            
            for mol in reactant_mols:
                reactant_atoms.update([atom.GetSymbol() for atom in mol.GetAtoms()])
            
            for mol in product_mols:
                product_atoms.update([atom.GetSymbol() for atom in mol.GetAtoms()])
            
            halogen_lost = any(halogen in reactant_atoms and halogen not in product_atoms 
                             for halogen in ['I', 'Br', 'Cl'])
            
            # Check for alkyne presence in both reactants and products
            alkyne_pattern = Chem.MolFromSmarts("[C]#[C]")
            has_alkyne_reactant = any(mol.HasSubstructMatch(alkyne_pattern) for mol in reactant_mols)
            has_alkyne_product = any(mol.HasSubstructMatch(alkyne_pattern) for mol in product_mols)
            
            return halogen_lost and has_alkyne_reactant and has_alkyne_product
            
        except Exception:
            return False
