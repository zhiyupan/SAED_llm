from saed.data.data import load_table_list, load_table, load_tables, load_labels

__all__ = ['load_table_list', 'load_table', 'load_tables', 'load_labels']

if __name__ == '__main__':
    from omegaconf import DictConfig
    config = DictConfig({'data': {'tables': {'path': 'data/tables'}, 'labels': {'path': 'data/labels'}}})
    df_table_list = load_table_list(config=config)
    dict_tables = load_tables(config=config)
    # df_labels = load_labels(config=config)
    print(df_table_list.head())