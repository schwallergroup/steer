"""Generated evaluation code for: Direct C-H arylation without pre-functionalization"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class DirectCHArylation(BaseScoring):
    """
    Evaluates routes for the presence of direct C-H arylation reactions
    that avoid pre-functionalization steps like halogenation.
    """
    
    def __init__(self, config: Dict):
        self.avoiding_prefunctionalization = config.get("avoiding_prefunctionalization", True)
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Direct C-H arylation not found
        else:
            return 1 - x  # Earlier occurrence is better (more strategic)
    
    def hit_condition(self, d) -> bool:
        """
        Detects direct C-H arylation by looking for:
        1. Formation of C-C bonds between aromatic systems
        2. No halogen leaving groups involved
        3. Presence of typical C-H activation patterns
        """
        metadata = d.get("metadata", {})
        if "mapped_reaction_smiles" not in metadata:
            return False
            
        rxn_smiles = metadata["mapped_reaction_smiles"]
        rxn_parts = rxn_smiles.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        products = rxn_parts[0]
        reactants = rxn_parts[1]
        
        try:
            prod_mol = Chem.MolFromSmiles(products)
            react_mols = [Chem.MolFromSmiles(r) for r in reactants.split(".")]
            
            if not prod_mol or not all(react_mols):
                return False
                
            # Check for direct C-H arylation patterns
            return self._is_direct_ch_arylation(prod_mol, react_mols)
            
        except Exception:
            return False
    
    def _is_direct_ch_arylation(self, product, reactants) -> bool:
        """
        Identifies direct C-H arylation by checking:
        1. New aromatic C-C bond formation
        2. Absence of halogen leaving groups
        3. Presence of aromatic C-H activation substrates
        """
        # Filter out small molecules (catalysts, bases, solvents)
        organic_reactants = [mol for mol in reactants if mol.GetNumAtoms() > 3]
        
        if len(organic_reactants) < 2:
            return False
            
        # Check for absence of halogens in reactants (avoiding pre-functionalized substrates)
        if self.avoiding_prefunctionalization:
            halogen_pattern = Chem.MolFromSmarts("[F,Cl,Br,I]")
            for reactant in organic_reactants:
                if reactant.HasSubstructMatch(halogen_pattern):
                    return False
        
        # Look for aromatic systems in reactants that could undergo C-H activation
        aromatic_patterns = [
            "[cH]",  # Aromatic C-H
            "c1ccccc1",  # Benzene ring
            "c1ccsc1",   # Thiophene
            "c1ccnc1",   # Pyridine-like
            "c1cscn1",   # Thiazole-like
        ]
        
        aromatic_reactants = 0
        for reactant in organic_reactants:
            for pattern_smarts in aromatic_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if reactant.HasSubstructMatch(pattern):
                    aromatic_reactants += 1
                    break
        
        # Need at least 2 aromatic reactants for arylation
        if aromatic_reactants < 2:
            return False
            
        # Check for new aromatic C-C bond formation
        # This is indicated by increased aromatic connectivity in product
        return self._has_new_aromatic_cc_bond(product, organic_reactants)
    
    def _has_new_aromatic_cc_bond(self, product, reactants) -> bool:
        """
        Checks if the product has aromatic C-C bonds not present in reactants
        """
        # Count aromatic C-C bonds in product
        prod_aromatic_cc = 0
        for bond in product.GetBonds():
            if (bond.GetBeginAtom().GetIsAromatic() and 
                bond.GetEndAtom().GetIsAromatic() and
                bond.GetBeginAtom().GetSymbol() == 'C' and
                bond.GetEndAtom().GetSymbol() == 'C'):
                prod_aromatic_cc += 1
        
        # Count aromatic C-C bonds in reactants
        react_aromatic_cc = 0
        for reactant in reactants:
            for bond in reactant.GetBonds():
                if (bond.GetBeginAtom().GetIsAromatic() and 
                    bond.GetEndAtom().GetIsAromatic() and
                    bond.GetBeginAtom().GetSymbol() == 'C' and
                    bond.GetEndAtom().GetSymbol() == 'C'):
                    react_aromatic_cc += 1
        
        # New aromatic C-C bond formation indicates C-H arylation
        return prod_aromatic_cc > react_aromatic_cc
