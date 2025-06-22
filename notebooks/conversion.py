import networkx as nx
from pydantic import BaseModel
from typing import Any, List

class TreeExport(BaseModel):
    rxn_list: List[str]
    graph: Any = None

    # Exporting
    def export(self):
        head = self.create_graph(self.rxn_list)

        json = {}
        # For each reachable subgraph from source node, serialize it into JSON
        smiles = self.graph.nodes[head]["attr"].get("smiles")
        j = {
            "smiles": smiles,
            "type": "mol",
            "in_stock": False,
            "children": self.json_serialize(self.graph, key=head),
        }
        return j

    def create_graph(self, rxns):
        dg = nx.DiGraph()
        for r in rxns:
            rcts, prd = r.split('>>')
            dg.add_node(prd, attr={"smiles":prd})
            for rct in rcts.split('.'):
                dg.add_node(rct, attr={"smiles":rct})
                dg.add_edge(prd, rct)

        # Get head node
        head = [n for n in dg.nodes if dg.in_degree(n) == 0][0]
        self.graph = dg
        return head

    def json_serialize(self, G, key="10"):
        """Serialize a single reachable subgraph from source node into JSON."""

        json = {}
        successors = G.successors(key)
        slist = []
        for s in successors:
            props = G.nodes[s]
            if len(list(G.successors(s))) > 0:
                # Get properties of the node
                if "attr" not in props.keys():
                    continue
                if "smiles" in props["attr"].keys():
                    smiles = props["attr"]["smiles"]

                # Format json
                try: 
                    slist.append(
                        {
                            "smiles": smiles,
                            "type": "mol",
                            "in_stock": False,
                            "children": self.json_serialize(G, key=s)
                        }
                    )
                except:
                    slist.append(
                        {
                            "smiles": smiles,
                            "type": "mol",
                            "in_stock": False,
                        }
                    )
            else:
                slist.append(
                    {
                        "smiles": s,
                        "type": "mol",
                        "in_stock": False,
                    }
                )

        final_json = [{"smiles": "", "type": "reaction", "children": slist}]
        return final_json

    

class ReaxysConvert(BaseModel):
    def from_reaxys(self, rxs_doc):
        all_routes = {}
        jobs = rxs_doc['result']['jobs']
        for job in jobs:
            if job:
                routes = job['results']['rxspm:hasRoute']
                for r, rte in enumerate(routes):
                    #  TODO getting an error here
                    ##################

                    steps = rte['rxspm:hasStep']
                    all_routes[r] = []
                    for step in steps:
                        rxnj = step['rxspm:hasReaction']
                        rxnsmi = self.get_rxn_smiles(rxnj)
                        all_routes[r].append(rxnsmi)
        
        trees = []
        for r, rte in all_routes.items():
            tree = TreeExport(rxn_list=rte).export()
            trees.append(tree)
        return trees

    def get_rxn_smiles(self, rxnj):
        prods = rxnj['rxspm:hasProduct']
        rcts = rxnj['rxspm:hasStartingMaterial']
        rxn = ""
        for r in rcts:
            rxn += r['rxspm:hasSubstance']['edm:smiles'] + '.'
        rxn = rxn.strip('.') + '>>'
        for p in prods:
            rxn += p['rxspm:hasSubstance']['edm:smiles'] + '.'
        rxn = rxn.strip('.')
        return rxn