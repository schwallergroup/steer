"""Generated evaluation code for: Late intramolecular Negishi cyclization for ring closure"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateNegishiCyclization(BaseScoring):
    """
    Evaluates routes for late-stage intramolecular Negishi cyclization that forms aromatic rings.
    
    Checks for the presence of intramolecular Negishi coupling reactions that form aromatic
    rings, preferring when these occur late in the synthesis (closer to the final product).
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "float")
        self.target_depth = config.get("target_depth", {}).get("value", 0.2)  # Late stage preferred
        
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Reaction doesn't occur
        else:
            # Late-stage cyclization is better (lower depth fraction)
            if self.condition_type == "bool":
                return 1 if x <= self.target_depth else 0
            else:
                # Penalize early cyclization, reward late cyclization
                return max(0, 1 - x)
    
    def hit_condition(self, d) -> bool:
        """Check if reaction is an intramolecular Negishi cyclization forming aromatic ring"""
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn:
            return False
            
        try:
            rxn_parts = mapped_rxn.split(">>")
            if len(rxn_parts) != 2:
                return False
                
            product_smiles = rxn_parts[0]
            reactants_smiles = rxn_parts[1].split(".")
            
            product_mol = Chem.MolFromSmiles(product_smiles)
            reactant_mols = [Chem.MolFromSmiles(r) for r in reactants_smiles if Chem.MolFromSmiles(r)]
            
            if not product_mol or not reactant_mols:
                return False
            
            # Check for intramolecular reaction (same carbon skeleton in main reactant and product)
            if not self._is_intramolecular(product_mol, reactant_mols):
                return False
                
            # Check for Negishi coupling pattern (C-Zn bond formation with aromatic C)
            if not self._is_negishi_coupling(product_mol, reactant_mols):
                return False
                
            # Check if aromatic ring is formed
            if not self._forms_aromatic_ring(product_mol, reactant_mols):
                return False
                
            return True
            
        except Exception:
            return False
    
    def _is_intramolecular(self, product_mol, reactant_mols):
        """Check if reaction is intramolecular by comparing atom map numbers"""
        # Get main organic reactant (largest molecule without metals)
        main_reactant = None
        for mol in reactant_mols:
            if mol.GetNumAtoms() > 5:  # Skip small molecules like ZnX2
                has_metal = any(atom.GetSymbol() in ['Zn', 'Pd', 'Ni'] for atom in mol.GetAtoms())
                if not has_metal or mol.GetNumAtoms() > 10:  # Main substrate
                    main_reactant = mol
                    break
        
        if not main_reactant:
            return False
            
        # Check if both coupling carbons are in the same reactant molecule
        product_maps = {atom.GetAtomMapNum() for atom in product_mol.GetAtoms() if atom.GetAtomMapNum() > 0}
        reactant_maps = {atom.GetAtomMapNum() for atom in main_reactant.GetAtoms() if atom.GetAtomMapNum() > 0}
        
        return len(product_maps.intersection(reactant_maps)) >= 2
    
    def _is_negishi_coupling(self, product_mol, reactant_mols):
        """Check for Negishi coupling pattern"""
        # Look for zinc-containing reactant
        has_zinc = any(
            any(atom.GetSymbol() == 'Zn' for atom in mol.GetAtoms()) 
            for mol in reactant_mols
        )
        
        # Look for palladium catalyst (may not always be mapped)
        has_pd_catalyst = any(
            any(atom.GetSymbol() in ['Pd', 'Ni'] for atom in mol.GetAtoms())
            for mol in reactant_mols
        )
        
        # Check for C-C bond formation pattern typical of cross-coupling
        if not has_zinc:
            return False
            
        # Additional check: look for halide leaving group in reactants but not product
        halogen_pattern = Chem.MolFromSmarts('[#6][Cl,Br,I]')
        reactant_has_halogen = any(mol.HasSubstructMatch(halogen_pattern) for mol in reactant_mols)
        product_has_halogen = product_mol.HasSubstructMatch(halogen_pattern)
        
        return reactant_has_halogen and not product_has_halogen
    
    def _forms_aromatic_ring(self, product_mol, reactant_mols):
        """Check if an aromatic ring is formed in the reaction"""
        # Count aromatic rings in product vs main reactant
        main_reactant = max(reactant_mols, key=lambda x: x.GetNumAtoms())
        
        product_aromatic_rings = self._count_aromatic_rings(product_mol)
        reactant_aromatic_rings = self._count_aromatic_rings(main_reactant)
        
        # New aromatic ring should be formed
        return product_aromatic_rings > reactant_aromatic_rings
    
    def _count_aromatic_rings(self, mol):
        """Count number of aromatic rings in molecule"""
        ring_info = mol.GetRingInfo()
        aromatic_rings = 0
        
        for ring in ring_info.AtomRings():
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                aromatic_rings += 1
                
        return aromatic_rings
