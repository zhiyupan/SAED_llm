import os 
import os.path as osp
import json

import hydra
from omegaconf import DictConfig, OmegaConf

from saed.onto import OntologyDAG
from saed.data import load_labels
from saed.utils import path_level_f1_precision_recall, node_level_f1_precision_recall

here = osp.dirname(osp.abspath(__file__))
root_path = osp.join(here, "..", "..")

@hydra.main(version_base=None, config_path=osp.join(root_path, "config"), config_name="config")
def main(cfg: DictConfig):
    results_dir = osp.join(root_path, cfg['experiments']['outputs']['results']['dir'], cfg['llms']['name'])
    
    if not osp.exists(results_dir):
        raise FileExistsError(f"Results directory {results_dir}  does not exist")
    
    # Load the data
    df_labels = load_labels(cfg)
    
    # Load the ontology
    ontology_dag = OntologyDAG(cfg)
    ontology_dag.build_dag()
    
    # Load predictions
    with open(osp.join(results_dir, 'predictions.json')) as f:
        results_pred = json.load(f)
        
    eval_json = []
    for r in results_pred:
        table_id, table_name, column_id, column_name = r["table_id"], r["table_name"], r["column_id"], r["column_name"]
        label_c1l1 = df_labels[(df_labels["table_id"] == table_id) & (df_labels["column_id"] == column_id)]["class1_level1_name"].values[0]
        label_c1l2 = df_labels[(df_labels["table_id"] == table_id) & (df_labels["column_id"] == column_id)]["class1_level2_name"].values[0]
        label_c2l1 = df_labels[(df_labels["table_id"] == table_id) & (df_labels["column_id"] == column_id)]["class2_level1_name"].values[0]
        label_c2l2 = df_labels[(df_labels["table_id"] == table_id) & (df_labels["column_id"] == column_id)]["class2_level2_name"].values[0]
        pred_paths = []
        for path in r["paths"]:
            pred_path = []
            for o in path:
                pred_path.append(ontology_dag.nodes[o].name)
            pred_paths.append(pred_path)

        gt_paths = []
        if label_c1l1 != "-":
            gt_path = []
            gt_path.append(label_c1l1)
            if label_c1l2 != "-":
                gt_path.append(label_c1l2)
            gt_paths.append(pred_path)
        if label_c2l1 != "-":
            gt_path = []
            gt_path.append(label_c2l1)
            if label_c2l2 != "-":
                gt_path.append(label_c2l2)
            gt_paths.append(pred_path)

        eval_json.append({
            "table_id": table_id,
            "table_name": table_name,
            "column_id": column_id,
            "column_name": column_name,
            "pred_paths": pred_paths,
            "gt_paths": gt_paths
        })
    
    with open(osp.join(results_dir, 'results.json'), 'w') as outfile:
        json.dump(eval_json, outfile)
    
    path_macro_precision, path_macro_recall, path_macro_f1, path_micro_precision, path_micro_recall, path_micro_f1 = path_level_f1_precision_recall(eval_json)
    node_macro_precision, node_macro_recall, node_macro_f1, node_micro_precision, node_micro_recall, node_micro_f1 = node_level_f1_precision_recall(eval_json)
    
    print(f"+++++++++++++++++Node Level+++++++++++++++++")
    print(f"\tMacro Precision: {path_macro_precision}")
    print(f"\tMacro Recall: {path_macro_recall}")
    print(f"\tMacro F1: {path_macro_f1}")
    print(f"\tMicro Precision: {path_micro_precision}")
    print(f"\tMicro Recall: {path_micro_recall}")
    print(f"\tMicro F1: {path_micro_f1}")
    print(f"+++++++++++++++++Node Level+++++++++++++++++")
    print(f"\tMacro Precision: {node_macro_precision}")
    print(f"\tMacro Recall: {node_macro_recall}")
    print(f"\tMacro F1: {node_macro_f1}")
    print(f"\tMicro Precision: {node_micro_precision}")
    print(f"\tMicro Recall: {node_micro_recall}")
    print(f"\tMicro F1: {node_micro_f1}")
    
    with open(osp.join(results_dir, 'results.txt'), 'a') as f:
        f.write(f"+++++++++++++++++Node Level+++++++++++++++++\n")
        f.write(f"\tMacro Precision: {path_macro_precision}\n")
        f.write(f"\tMacro Recall: {path_macro_recall}\n")
        f.write(f"\tMacro F1: {path_macro_f1}\n")
        f.write(f"\tMicro Precision: {path_micro_precision}\n")
        f.write(f"\tMicro Recall: {path_micro_recall}\n")
        f.write(f"\tMicro F1: {path_micro_f1}\n")
        
    with open(osp.join(results_dir, 'results.txt'), 'a') as f:
        f.write(f"+++++++++++++++++Node Level+++++++++++++++++\n")
        f.write(f"\tMacro Precision: {node_macro_precision}\n")
        f.write(f"\tMacro Recall: {node_macro_recall}\n")
        f.write(f"\tMacro F1: {node_macro_f1}\n")
        f.write(f"\tMicro Precision: {node_micro_precision}\n")
        f.write(f"\tMicro Recall: {node_micro_recall}\n")
        f.write(f"\tMicro F1: {node_micro_f1}\n")

@hydra.main(version_base=None, config_path=osp.join(root_path, "config"), config_name="config")
def print_results(cfg: DictConfig):
    
    results_dir = osp.join(root_path, cfg['experiments']['outputs']['results']['dir'], cfg['llms']['name'])
    if not osp.exists(results_dir):
        raise FileExistsError(f"Results directory {results_dir}  does not exist")
    
    results_json_file = osp.join(results_dir, 'results.json')
    with open(results_json_file, 'r', encoding='utf-8') as f:
        results_json = json.load(f)
    path_macro_precision, path_macro_recall, path_macro_f1, path_micro_precision, path_micro_recall, path_micro_f1 = path_level_f1_precision_recall(results_json)
    node_macro_precision, node_macro_recall, node_macro_f1, node_micro_precision, node_micro_recall, node_micro_f1 = node_level_f1_precision_recall(results_json)
    print(f"+++++++++++++++++Node Level+++++++++++++++++")
    print(f"\tMacro Precision: {path_macro_precision}")
    print(f"\tMacro Recall: {path_macro_recall}")
    print(f"\tMacro F1: {path_macro_f1}")
    print(f"\tMicro Precision: {path_micro_precision}")
    print(f"\tMicro Recall: {path_micro_recall}")
    print(f"\tMicro F1: {path_micro_f1}")
    print(f"+++++++++++++++++Node Level+++++++++++++++++")
    print(f"\tMacro Precision: {node_macro_precision}")
    print(f"\tMacro Recall: {node_macro_recall}")
    print(f"\tMacro F1: {node_macro_f1}")
    print(f"\tMicro Precision: {node_micro_precision}")
    print(f"\tMicro Recall: {node_micro_recall}")
    print(f"\tMicro F1: {node_micro_f1}")

if __name__ == "__main__":
    main()
    # print_results()