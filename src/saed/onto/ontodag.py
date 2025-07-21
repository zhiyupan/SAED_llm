import os
import os.path as osp
from collections import defaultdict
import owlready2
from owlready2 import Thing
from omegaconf import DictConfig
from saed.onto.ontoclass import OntologyClass

here = osp.dirname(osp.abspath(__file__))
root_path = osp.join(here, "../../../")
# default_rdf_file_path = osp.join(here, "../../../data/ontology/BEO_clean.rdf")

class OntologyDAG:
    def __init__(self, config: DictConfig):
        """
        Represents the ontology as a Directed Acyclic Graph (DAG).

        Attributes:
            nodes (dict): A dictionary to store nodes with URL as keys and OntologyClass instances as values.
            edges (defaultdict): A dictionary to store edges, representing subclass relationships (parent to children).
        """
        self.rdf_file_path = osp.join(root_path, config["data"]["ontology"]["path"])
        self.nodes = {}  # Maps URL to OntologyClass
        self.edges_subclassof = defaultdict(list)  # Maps URL to list of child URLs
        self.edges = defaultdict(list)
        self.root = None

    def build_dag(self, rdf_file_path=None):
        """
        Builds the ontology DAG from an RDF file.

        Args:
            rdf_file_path (str): The path to the RDF file containing the ontology.
        """
        if rdf_file_path is not None:
            self.rdf_file_path = rdf_file_path

        onto = owlready2.get_ontology(self.rdf_file_path).load()
        for cls in onto.classes():
            url = cls.iri
            self.nodes[url] = OntologyClass(url, name=cls.name, label=cls.label, comment=cls.comment)
            for parent in cls.subclasses():
                if parent.iri == url:
                    continue
                if self.edges_subclassof.get(parent.iri) is None:
                    self.edges_subclassof[parent.iri] = []
                self.edges_subclassof[parent.iri].append(url)
            
        self.edges = {k: [] for k in self.nodes}
        for parent, children in self.edges_subclassof.items():
            for child in children:
                self.edges[child].append(parent)
        self.root = Thing.iri
        
    def __repr__(self):
        return f"OntologyDAG(nodes={list(self.nodes.keys())}, edges={dict(self.edges)})"

# Example usage
if __name__ == "__main__":
    # Create an ontology DAG
    config = DictConfig({ "data": { "ontology": { "path": "data/ontology/BEO_clean.rdf" } } })
    ontology_dag = OntologyDAG(config=config)

    # Build the DAG from an RDF file
    ontology_dag.build_dag()

    # Print the ontology DAG
    print(ontology_dag)
