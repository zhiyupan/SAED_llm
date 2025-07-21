import pandas as pd
from collections import deque
from saed.ensemble import DecisionMaker

def dataframe_to_markdown(df: pd.DataFrame, k: int=5):
    
    subset = df.head(k)
    headers = "| " + " | ".join(subset.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(subset.columns)) + " |"

    rows = []
    for _, row in subset.iterrows():
        row_str = "| " + " | ".join(map(str, row.values)) + " |"
        rows.append(row_str)

    md_table = headers + "\n" + sep + "\n" + "\n".join(rows)
    return md_table


def bfs_search(table_name, table_in_markdown, column_name, ontology_dag, decision_maker,  max_depth):
    queue = deque()
    queue.append((0, ontology_dag.root, []))
    possible_path = []
    while queue:
        level, parent_level_ontology_class, search_path = queue.popleft()
        if level >= max_depth:
            possible_path.append(search_path)
            continue
        if len(ontology_dag.edges[parent_level_ontology_class]) == 0:
            possible_path.append(search_path)
            continue
        current_level_ontology_classes = [ontology_dag.nodes[o].name for o in ontology_dag.edges[parent_level_ontology_class]]
        current_level_ontology_classes_url_dict = {ontology_dag.nodes[o].name:o for o in ontology_dag.edges[parent_level_ontology_class]}
        result = decision_maker.decision_making(table_name, table_in_markdown, column_name, current_level_ontology_classes)
        if  result == "-":
            print("\tNone")
            possible_path.append(search_path)
            continue
        else:
            selected_ontology_classes = result.split(", ")
            for selected_ontology_class in selected_ontology_classes:
                if selected_ontology_class in current_level_ontology_classes:
                    print(f"\t{selected_ontology_class}")
                    queue.append((level + 1, current_level_ontology_classes_url_dict[selected_ontology_class], search_path+[current_level_ontology_classes_url_dict[selected_ontology_class]]))
                else:
                    print(f"Error: {selected_ontology_class} is not in current level ontology classes.")
    return possible_path


def path_level_f1_precision_recall(data):
    """
    Calculate micro and macro precision, recall, and F1
    """
    precisions = []
    recalls = []
    f1s = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for row in data:
        gt_set = {tuple(p) for p in row['gt_paths']}
        pred_set = {tuple(p) for p in row['pred_paths']}
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        # print("\tPrecision:", precision, "Recall:", recall, "F1:", f1)
        # print("\tTP:", tp, "FP:", fp, "FN:", fn)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    # print("Total TP:", total_tp, "Total FP:", total_fp, "Total FN:", total_fn)
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


def flatten_list_to_set(paths):
    """
    Given a list of paths (each a list of nodes), flatten all nodes 
    into a single tuple. 
    Examples:
    [['TemporalEntity', 'Interval']] -> ('TemporalEntity', 'Interval')
    [['TemporalEntity', 'Interval'], ['TemporalEntity', 'TemporalPosition']] -> ('TemporalEntity', 'Interval', 'TemporalPosition')
    """
    all_nodes = []
    for p in paths:
        all_nodes.extend(p)
    return set(all_nodes)

def node_level_f1_precision_recall(data):
    """
    Calculate micro and macro precision, recall, and F1 at the node level
    """
    precisions = []
    recalls = []
    f1s = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for row in data:
        gt_set = flatten_list_to_set(row['gt_paths'])
        pred_set = flatten_list_to_set(row['pred_paths'])
        tp = len(pred_set.intersection(gt_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        # print("\tPrecision:", precision, "Recall:", recall, "F1:", f1)
        # print("\tTP:", tp, "FP:", fp, "FN:", fn)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    # print("Total TP:", total_tp, "Total FP:", total_fp, "Total FN:", total_fn)
    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1