"""Generated evaluation code for: Early stage amide dehydration to nitrile"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class EarlyAmideDehydration(BaseScoring):
    """
    Evaluates synthesis routes for early stage amide dehydration to nitrile reactions.
    
    Checks if primary amides are converted to nitriles early in the synthesis sequence,
    which establishes stable intermediates for subsequent transformations.
    """
    
    def __init__(self, config: Dict):
        self.stage_preference = config.get("stage", "early")
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Early stage is preferred (lower depth fraction is better)
            return 1 - x
    
    def hit_condition(self, d):
        """Check if this reaction node represents amide dehydration to nitrile."""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            reactants = rxn_parts[0]
            products = rxn_parts[1]
            
            # Parse reactants and products
            reactant_mols = []
            for smi in reactants.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    reactant_mols.append(mol)
            
            product_mols = []
            for smi in products.split("."):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    product_mols.append(mol)
            
            if not reactant_mols or not product_mols:
                return False
            
            # Define SMARTS patterns
            primary_amide_pattern = "[C](=[O])[NH2]"  # Primary amide
            nitrile_pattern = "[C]#[N]"  # Nitrile group
            
            primary_amide_smarts = Chem.MolFromSmarts(primary_amide_pattern)
            nitrile_smarts = Chem.MolFromSmarts(nitrile_pattern)
            
            # Check if any reactant contains primary amide
            has_reactant_amide = False
            for mol in reactant_mols:
                if mol.HasSubstructMatch(primary_amide_smarts):
                    has_reactant_amide = True
                    break
            
            # Check if any product contains nitrile
            has_product_nitrile = False
            for mol in product_mols:
                if mol.HasSubstructMatch(nitrile_smarts):
                    has_product_nitrile = True
                    break
            
            # Verify this is an amide dehydration by checking atom mapping
            if has_reactant_amide and has_product_nitrile:
                return self._verify_amide_to_nitrile_transformation(reactant_mols, product_mols)
            
            return False
            
        except Exception:
            return False
    
    def _verify_amide_to_nitrile_transformation(self, reactant_mols, product_mols):
        """
        Verify that the same carbon atom that was part of the amide carbonyl
        becomes part of the nitrile group using atom mapping.
        """
        try:
            # Find mapped atoms in amide groups in reactants
            amide_carbons = set()
            for mol in reactant_mols:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0 and atom.GetSymbol() == 'C':
                        # Check if this carbon is part of an amide
                        neighbors = [n for n in atom.GetNeighbors()]
                        has_carbonyl_o = any(n.GetSymbol() == 'O' and 
                                           any(b.GetBondType() == Chem.BondType.DOUBLE 
                                               for b in n.GetBonds()) 
                                           for n in neighbors)
                        has_amine_n = any(n.GetSymbol() == 'N' and 
                                        len([nb for nb in n.GetNeighbors()]) <= 2
                                        for n in neighbors)
                        
                        if has_carbonyl_o and has_amine_n:
                            amide_carbons.add(atom.GetAtomMapNum())
            
            # Find mapped atoms in nitrile groups in products
            nitrile_carbons = set()
            for mol in product_mols:
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 0 and atom.GetSymbol() == 'C':
                        # Check if this carbon is part of a nitrile
                        neighbors = [n for n in atom.GetNeighbors()]
                        has_triple_n = any(n.GetSymbol() == 'N' and 
                                         any(b.GetBondType() == Chem.BondType.TRIPLE 
                                             for b in atom.GetBonds() if b.GetOtherAtom(atom) == n)
                                         for n in neighbors)
                        
                        if has_triple_n:
                            nitrile_carbons.add(atom.GetAtomMapNum())
            
            # Check if any amide carbon became a nitrile carbon
            return bool(amide_carbons.intersection(nitrile_carbons))
            
        except Exception:
            # Fall back to simple presence check if mapping verification fails
            return True
