"""Generated evaluation code for: Late stage nucleophilic aromatic substitution"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageNucleophilicAromaticSubstitution(BaseScoring):
    """
    Evaluates synthesis routes for late-stage nucleophilic aromatic substitution reactions.
    
    This scorer identifies SNAr reactions where a nucleophile substitutes an aromatic leaving group,
    and rewards routes where this occurs in the final steps of synthesis.
    """
    
    def __init__(self, config: Dict):
        self.timing = config.get("timing", "late")  # "late", "early", or "any"
        self.step_position = config.get("step_position", "final")  # "final" or "any"
    
    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # SNAr reaction doesn't occur
        
        if self.timing == "late" or self.step_position == "final":
            # Reward later occurrence (lower depth fraction is better for late-stage)
            return 10 * (1 - x)  # x is depth fraction, so 1-x rewards late-stage
        elif self.timing == "early":
            # Reward earlier occurrence
            return 10 * x
        else:  # timing == "any"
            return 10  # Just presence matters
    
    def hit_condition(self, d) -> bool:
        """
        Detects nucleophilic aromatic substitution reactions by looking for:
        1. Aromatic ring with electron-withdrawing groups
        2. Loss of a leaving group (F, Cl, Br, I, NO2, etc.)
        3. Addition of a nucleophile
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
        
        try:
            reactants_smiles, products_smiles = mapped_rxn.split(">>")
            
            # Parse molecules
            product = Chem.MolFromSmiles(products_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".") if r.strip()]
            
            if not product or not reactants:
                return False
            
            # Look for aromatic substrate in reactants
            aromatic_substrate = None
            nucleophile = None
            
            for reactant in reactants:
                if self._has_activated_aromatic_ring(reactant):
                    aromatic_substrate = reactant
                elif self._is_nucleophile(reactant):
                    nucleophile = reactant
            
            if not aromatic_substrate:
                return False
            
            # Check if substitution occurred on aromatic ring
            return self._detect_aromatic_substitution(aromatic_substrate, product)
            
        except Exception:
            return False
    
    def _has_activated_aromatic_ring(self, mol) -> bool:
        """Check if molecule has an aromatic ring with electron-withdrawing groups and leaving groups"""
        if not mol:
            return False
        
        # Patterns for activated aromatic rings with leaving groups
        activated_aromatic_patterns = [
            # Halogen substituted aromatics with EWG
            "[cH0:1]([F,Cl,Br,I])c1ccc([N+](=O)[O-])cc1",  # para-nitro haloarene
            "[cH0:1]([F,Cl,Br,I])c1cc([N+](=O)[O-])ccc1",   # meta-nitro haloarene
            "[cH0:1]([F,Cl,Br,I])c1ccc(C(=O)[O,C,N])cc1",   # para-carbonyl haloarene
            "[cH0:1]([F,Cl,Br,I])c1cc(C(=O)[O,C,N])ccc1",   # meta-carbonyl haloarene
            "[cH0:1]([F,Cl,Br,I])c1ccc([C,S](=O)(=O)[O,N,C])cc1",  # para-sulfonyl haloarene
            "[cH0:1]([F,Cl,Br,I])c1ccncc1",  # pyridine halide
            "[cH0:1]([F,Cl,Br,I])c1ncccn1",  # pyrimidine halide
        ]
        
        for pattern in activated_aromatic_patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        
        return False
    
    def _is_nucleophile(self, mol) -> bool:
        """Check if molecule could act as a nucleophile in SNAr"""
        if not mol:
            return False
        
        # Common nucleophile patterns for SNAr
        nucleophile_patterns = [
            "[NH2,NH1,NH0;!$(N-[S,P](=O))]",  # Amines (not sulfonamides/phosphoramides)
            "[OH;!$(O-S(=O)=O)]",             # Alcohols/phenols (not sulfates)
            "[SH,S-]",                         # Thiols/thiolates
            "N1CCNCC1",                       # Piperazine
            "N1CCOCC1",                       # Morpholine
            "N1CCCCC1",                       # Piperidine
            "[O-,S-]",                        # Alkoxides/thiolates
        ]
        
        for pattern in nucleophile_patterns:
            patt_mol = Chem.MolFromSmarts(pattern)
            if patt_mol and mol.HasSubstructMatch(patt_mol):
                return True
        
        return False
    
    def _detect_aromatic_substitution(self, reactant, product) -> bool:
        """Check if aromatic substitution occurred by comparing reactant and product"""
        # Look for loss of leaving group and gain of nucleophilic group
        
        # Count aromatic carbons with heteroatom substituents
        aromatic_substituted_reactant = len(reactant.GetSubstructMatches(
            Chem.MolFromSmarts("[c][F,Cl,Br,I,N,O,S]")
        ))
        
        aromatic_substituted_product = len(product.GetSubstructMatches(
            Chem.MolFromSmarts("[c][N,O,S;!$(N(=O)=O)]")  # Exclude nitro groups
        ))
        
        # Check if we have maintained aromatic ring but changed substitution pattern
        reactant_aromatic_atoms = len([a for a in reactant.GetAtoms() if a.GetIsAromatic()])
        product_aromatic_atoms = len([a for a in product.GetAtoms() if a.GetIsAromatic()])
        
        # Aromatic ring should be preserved, substitution pattern should change
        return (reactant_aromatic_atoms > 0 and 
                abs(reactant_aromatic_atoms - product_aromatic_atoms) <= 2 and
                aromatic_substituted_product >= aromatic_substituted_reactant)
