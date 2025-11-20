"""Generated evaluation code for: Late stage Suzuki coupling for biaryl formation"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageSuzuki(BaseScoring):
    """
    Evaluates routes for late-stage Suzuki coupling reactions that form biaryl bonds.
    Checks if a Suzuki-Miyaura coupling occurs and scores based on how late in the 
    synthesis it happens (later is better).
    """
    
    def __init__(self, config: Dict):
        self.bond_smarts = config["parameters"].get("bond_smarts", "c-c")
        self.context = config["parameters"].get("context", "biaryl")
        self.reaction_type = config["parameters"].get("reaction_type", "suzuki")
        self.timing = config["parameters"].get("timing", "late")

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Suzuki coupling doesn't happen
        else:
            # Late-stage coupling is better - return higher score for later reactions
            return 10 * (1 - x)  # x is depth fraction, so (1-x) gives higher scores for later reactions

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction is a Suzuki coupling forming a biaryl bond.
        """
        metadata = d.get("metadata", {})
        
        # Check if reaction involves Suzuki-like transformation
        if not self._is_suzuki_reaction(metadata):
            return False
            
        # Check if it forms a biaryl bond
        return self._forms_biaryl_bond(metadata)
    
    def _is_suzuki_reaction(self, metadata) -> bool:
        """
        Detect Suzuki coupling by looking for boronic acid/ester patterns
        and palladium-catalyzed C-C bond formation indicators.
        """
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
            
        try:
            reactants_smiles = mapped_rxn.split(">>")[0]
            
            # Look for boronic acid or boronic ester patterns in reactants
            boronic_patterns = [
                "B(O)O",  # Boronic acid
                "B1OC(C)(C)C(C)(C)O1",  # Pinacol boronate
                "[B-]([O-])([O-])[O-]",  # Borate
                "B(OC)OC"  # Methyl boronate
            ]
            
            for pattern in boronic_patterns:
                boronic_mol = Chem.MolFromSmarts(pattern)
                if boronic_mol:
                    reactant_mols = [Chem.MolFromSmiles(smi.strip()) for smi in reactants_smiles.split(".")]
                    for mol in reactant_mols:
                        if mol and mol.HasSubstructMatch(boronic_mol):
                            return True
                            
            return False
            
        except Exception:
            return False
    
    def _forms_biaryl_bond(self, metadata) -> bool:
        """
        Check if the reaction forms a new C-C bond between two aromatic rings.
        """
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            reactants_smiles = rxn_parts[0]
            product_smiles = rxn_parts[1]
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            if not product_mol:
                return False
                
            # Check for biaryl pattern in product
            biaryl_pattern = Chem.MolFromSmarts("c-c")  # Aromatic C-C bond
            if not product_mol.HasSubstructMatch(biaryl_pattern):
                return False
                
            # Verify this is a new bond by checking it doesn't exist in reactants
            reactant_mols = []
            for smi in reactants_smiles.split("."):
                mol = Chem.MolFromSmiles(smi.strip())
                if mol:
                    reactant_mols.append(mol)
            
            # Get atom mappings to track bond formation
            return self._verify_new_biaryl_formation(reactant_mols, product_mol)
            
        except Exception:
            return False
    
    def _verify_new_biaryl_formation(self, reactant_mols, product_mol) -> bool:
        """
        Verify that a new biaryl bond is formed by checking atom mappings.
        """
        try:
            # Get mapped atoms in product that form aromatic C-C bonds
            product_aromatic_bonds = []
            for bond in product_mol.GetBonds():
                if (bond.GetBeginAtom().GetIsAromatic() and 
                    bond.GetEndAtom().GetIsAromatic() and
                    bond.GetBeginAtom().GetAtomMapNum() > 0 and
                    bond.GetEndAtom().GetAtomMapNum() > 0):
                    
                    map1 = bond.GetBeginAtom().GetAtomMapNum()
                    map2 = bond.GetEndAtom().GetAtomMapNum()
                    product_aromatic_bonds.append((min(map1, map2), max(map1, map2)))
            
            # Check if any of these bonds are new (don't exist in reactants)
            for reactant in reactant_mols:
                reactant_aromatic_bonds = []
                for bond in reactant.GetBonds():
                    if (bond.GetBeginAtom().GetIsAromatic() and 
                        bond.GetEndAtom().GetIsAromatic() and
                        bond.GetBeginAtom().GetAtomMapNum() > 0 and
                        bond.GetEndAtom().GetAtomMapNum() > 0):
                        
                        map1 = bond.GetBeginAtom().GetAtomMapNum()
                        map2 = bond.GetEndAtom().GetAtomMapNum()
                        reactant_aromatic_bonds.append((min(map1, map2), max(map1, map2)))
                
                # If we find new aromatic C-C bonds in product, it's likely biaryl formation
                new_bonds = set(product_aromatic_bonds) - set(reactant_aromatic_bonds)
                if new_bonds:
                    return True
                    
            return False
            
        except Exception:
            return False
