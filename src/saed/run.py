import os
import os.path as osp
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf

from saed.ensemble import DecisionMaker
from saed.onto import OntologyDAG
from saed.data import load_table_list, load_tables, load_labels
from saed.utils import dataframe_to_markdown, bfs_search

here = osp.dirname(osp.abspath(__file__))
root_path = osp.join(here, "..", "..")

@hydra.main(version_base=None, config_path=osp.join(root_path, "config"), config_name="config")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    logs_dir = osp.join(root_path, cfg['experiments']['outputs']['logs']['dir'], cfg['llms']['name'])
    results_dir = osp.join(root_path, cfg['experiments']['outputs']['results']['dir'], cfg['llms']['name'])
    
    if osp.exists(logs_dir):
        shutil.rmtree(results_dir)
    os.makedirs(logs_dir, exist_ok=True)
    if osp.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the data
    df_table_list = load_table_list(cfg)
    dict_tables = load_tables(cfg)
    df_labels = load_labels(cfg)
    
    # Load the ontology
    ontology_dag = OntologyDAG(cfg)
    ontology_dag.build_dag()
    
    # Create the decision maker
    decision_maker = DecisionMaker(cfg)
    
    # Config parameters
    k = cfg['experiments']['k']
    max_depth = cfg['experiments']['max_depth']
    
    # Run the semantic annotation for each column in the tables
    semantic_annotation_predictions = []
    for index, row in df_labels.iterrows():
        table_id, column_name, column_id = row['table_id'], row['column_name'], int(row['column_id'])
        print('Table ID:', table_id, 'Column Name:', column_name)
        table = dict_tables[table_id]
        table_name = df_table_list[df_table_list["table_id"] == table_id]["table_name"].values[0]
        table_in_markdown = dataframe_to_markdown(table, k)
        paths= bfs_search(table_name, table_in_markdown, column_name, ontology_dag, decision_maker, max_depth)
        semantic_annotation_predictions.append(
            {
                "table_id": table_id,
                "table_name": table_name,
                "column_name": column_name,
                "column_id": column_id,
                "paths": paths
            }
        )
    # Convert the semantic_annotation_results to JSON format
    semantic_annotation_predictions_json = json.dumps(semantic_annotation_predictions, indent=4)
    # Save the JSON to a file
    with open(osp.join(results_dir, "predictions.json"), 'w') as f:
        f.write(semantic_annotation_predictions_json)
        
if __name__ == "__main__":
    main()