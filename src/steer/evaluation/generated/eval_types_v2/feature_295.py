"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzukiCoupling(BaseScoring):
    """
    Checks if a Suzuki coupling reaction forming a biaryl bond occurs late in the synthesis route.
    Detects the formation of C-C bonds between aromatic carbons via Suzuki coupling reaction.
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"]["bond_smarts"]
        self.reaction_type = config["parameters"]["reaction_type"]
        self.timing = config["parameters"]["timing"]
        
        # SMARTS pattern for Suzuki coupling detection
        # Boronic acid/ester + aryl halide -> biaryl
        self.suzuki_patterns = [
            "[c:1][B]([OH])[OH].[c:2][X]>>[c:1][c:2]",  # Boronic acid + aryl halide
            "[c:1][B]1OC(C)(C)CO1.[c:2][X]>>[c:1][c:2]",  # Boronic ester + aryl halide
            "[c:1][B]([OH])O.[c:2][X]>>[c:1][c:2]"  # Alternative boronic acid pattern
        ]
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # For late-stage reactions, higher depth fraction is better
            if self.timing == "late":
                return 10 * x  # Score 0-10, higher is better for later reactions
            else:
                return 10 * (1 - x)  # Early stage preferred
    
    def hit_condition(self, d) -> bool:
        """Check if this reaction node represents a Suzuki coupling forming a biaryl bond."""
        metadata = d.get("metadata", {})
        
        # Check if mapped reaction SMILES is available
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        mapped_rxn = metadata["mapped_reaction_smiles"]
        
        # Split reaction into reactants and products
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            products = [Chem.MolFromSmiles(p.strip()) for p in products_smiles.split(".")]
            
            # Filter out None molecules
            reactants = [r for r in reactants if r is not None]
            products = [p for p in products if p is not None]
            
            if not reactants or not products:
                return False
                
        except (ValueError, AttributeError):
            return False
        
        # Check if this looks like a Suzuki coupling
        if not self._is_suzuki_coupling(reactants, products):
            return False
            
        # Check if a biaryl bond (matching the bond_smarts pattern) is formed
        return self._forms_target_bond(reactants, products)
    
    def _is_suzuki_coupling(self, reactants, products) -> bool:
        """Check if the reaction pattern matches Suzuki coupling."""
        # Look for boron-containing reactant
        has_boron = any(mol.HasSubstructMatch(Chem.MolFromSmarts("[B]")) for mol in reactants)
        
        # Look for halogen-containing aromatic reactant
        halogen_patterns = ["[c][F]", "[c][Cl]", "[c][Br]", "[c][I]"]
        has_aryl_halide = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in halogen_patterns)
            for mol in reactants
        )
        
        # Alternative: check for boronic acid/ester patterns
        boronic_patterns = [
            "[c][B]([OH])[OH]",  # Boronic acid
            "[c][B]1OCC(C)(C)O1",  # Pinacol boronic ester
            "[c][B](O)O"  # Alternative boronic acid
        ]
        has_boronic = any(
            any(mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)) for pattern in boronic_patterns)
            for mol in reactants
        )
        
        return (has_boron or has_boronic) and has_aryl_halide
    
    def _forms_target_bond(self, reactants, products) -> bool:
        """Check if the target biaryl bond is formed in this reaction."""
        # Count biaryl bonds in reactants vs products
        bond_pattern = Chem.MolFromSmarts(self.bond_smarts)
        
        reactant_bonds = sum(len(mol.GetSubstructMatches(bond_pattern)) for mol in reactants)
        product_bonds = sum(len(mol.GetSubstructMatches(bond_pattern)) for mol in products)
        
        # Bond formation: more bonds in products than reactants
        return product_bonds > reactant_bonds
