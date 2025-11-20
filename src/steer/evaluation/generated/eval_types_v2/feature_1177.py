"""Generated evaluation code for: Late stage lactam reduction"""

from typing import Dict, Tuple
from rdkit import Chem
from steer.evaluation.synthesis.eval_types.base import BaseScoring
from steer.evaluation.synthesis.eval_types.multi_rxn import MultiRxnCondBase

class LateStageLatamReduction(BaseScoring):
    """
    Evaluates synthesis routes for late-stage lactam reduction reactions.
    Detects when a lactam (cyclic amide) is reduced to a diamine using
    strong reducing agents like LAH or borane complexes.
    """
    
    def __init__(self, config: Dict):
        self.condition_type = config.get("target_depth", {}).get("type", "fraction")
        self.target_depth = config.get("target_depth", {}).get("value", 0.8)
        
        # SMARTS patterns for lactam structures (common ring sizes)
        self.lactam_patterns = [
            "[#6]-1-[#6]-[#7](-[#1,#6])-[#6](=[#8])-[#6]-1",  # 5-membered lactam
            "[#6]-1-[#6]-[#6]-[#7](-[#1,#6])-[#6](=[#8])-[#6]-1",  # 6-membered lactam
            "[#6]-1-[#6]-[#6]-[#6]-[#7](-[#1,#6])-[#6](=[#8])-[#6]-1",  # 7-membered lactam
            "[#6]-1-[#6]-[#6]-[#6]-[#6]-[#7](-[#1,#6])-[#6](=[#8])-[#6]-1"  # 8-membered lactam
        ]
        
        # SMARTS patterns for corresponding diamines after reduction
        self.diamine_patterns = [
            "[#6]-1-[#6]-[#7](-[#1,#6])-[#6]-[#6]-[#6]-1",  # 5-membered diamine
            "[#6]-1-[#6]-[#6]-[#7](-[#1,#6])-[#6]-[#6]-[#6]-1",  # 6-membered diamine
            "[#6]-1-[#6]-[#6]-[#6]-[#7](-[#1,#6])-[#6]-[#6]-[#6]-1",  # 7-membered diamine
            "[#6]-1-[#6]-[#6]-[#6]-[#6]-[#7](-[#1,#6])-[#6]-[#6]-[#6]-1"  # 8-membered diamine
        ]

    def route_scoring(self, x) -> float:
        if x < 0:
            return 0  # Lactam reduction doesn't occur
        
        if self.condition_type == "bool":
            return 10 if x >= self.target_depth else 0
        else:
            # Late-stage is preferred, so higher depth fractions get higher scores
            if x >= self.target_depth:
                return 10
            else:
                return 10 * (x / self.target_depth)

    def hit_condition(self, d) -> bool:
        """
        Check if this reaction represents a lactam reduction.
        Looks for lactam in product and corresponding reduced form in reactants.
        """
        metadata = d.get("metadata", {})
        mapped_rxn = metadata.get("mapped_reaction_smiles", "")
        
        if not mapped_rxn or ">>" not in mapped_rxn:
            return False
            
        rxn_parts = mapped_rxn.split(">>")
        if len(rxn_parts) != 2:
            return False
            
        product_smiles = rxn_parts[0].strip()
        reactants_smiles = rxn_parts[1].strip()
        
        # Parse molecules
        try:
            product = Chem.MolFromSmiles(product_smiles)
            reactants = [Chem.MolFromSmiles(r.strip()) for r in reactants_smiles.split(".")]
            
            if not product or not all(reactants):
                return False
                
        except Exception:
            return False
        
        # Check if product contains lactam
        has_lactam = any(product.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                        for pattern in self.lactam_patterns)
        
        if not has_lactam:
            return False
            
        # Check if any reactant contains corresponding diamine structure
        has_diamine = False
        for reactant in reactants:
            if any(reactant.HasSubstructMatch(Chem.MolFromSmarts(pattern)) 
                   for pattern in self.diamine_patterns):
                has_diamine = True
                break
                
        # Additional check for presence of reducing agents in reactant names or SMILES
        has_reducing_agent = self._check_reducing_agents(reactants_smiles, metadata)
        
        return has_lactam and has_diamine and has_reducing_agent
    
    def _check_reducing_agents(self, reactants_smiles: str, metadata: Dict) -> bool:
        """
        Check for presence of common lactam reducing agents.
        """
        # Common reducing agents for lactam reduction
        reducing_agents = [
            "[Li+]",  # Lithium (for LAH)
            "[Al]",   # Aluminum (for LAH) 
            "[B]",    # Boron (for borane complexes)
            "B2H6",   # Diborane
            "BH3",    # Borane
        ]
        
        # Check SMILES for reducing agent patterns
        reactants_lower = reactants_smiles.lower()
        agent_keywords = ["lialh4", "lah", "bh3", "borane", "diborane", "lithium aluminum hydride"]
        
        # Check in SMILES or metadata for reducing agent indicators
        has_agent = (any(keyword in reactants_lower for keyword in agent_keywords) or
                    any(reactants_lower.count(agent.lower()) > 0 for agent in ["[Li+]", "[Al]", "[B]"]))
        
        # Also check reaction name/template if available
        reaction_name = metadata.get("template_name", "").lower()
        if reaction_name:
            has_agent = has_agent or any(keyword in reaction_name for keyword in agent_keywords)
            
        return has_agent
