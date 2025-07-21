# SAED-LLM: Semantic Annotation for Energy Data using ensemble decision-making with Large Language Models

This is official repo for "SAED-LLM: Semanti Annotatio for Energy Data using ensemble decision-making with Large Language Models".


## Python Environment Setup

1. conda environment
```
conda create --name saed python=3.11
conda activate saed
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
pip install -r requirements.txt
pip install -e .
```

## Datasets

We use the Building Energy Ontology (BEO) and related tabular data in the energy domain.


## Run experiments

Configure the project, for example, AZURE_OPENAI_API_KEY.

```
find config -name "*.yaml.example" -exec sh -c 'cp "$0" "${0%.yaml.example}.yaml"' {} \;
```

To run experiments, you can use the experiments using the following command:

```
python src/saed/run.py experiments=llm llms=azure_openai
```
Make sure to replace `experiments=llm` and `llms=azure_openai` with your specific experiment and LLM configurations if they differ. Currently, we have three experiment configurations, i.e., `llm`, `cot`, and `edm`, and two LLMs, i.e., `ollmama` and `azure_openai`. The models we used are `llama3.1:8b` and `gpt-4o-mini`. The output is a `predictions.json` file in the `outputs` directory, which contains a set of JSON objects and each JSON object has the following structure:

```json
{
 "table_id": "1.csv",
 "table_name": "example_table_name",
 "column_name": "example_column_name",
 "column_id": 0,
 "paths": [
 [
 "http://energy.linkeddata.es/em-kpi/ontology#EnergyConsumer"
 ],
 [
 "https://w3id.org/saref#Measurement",
 "https://w3id.org/saref#EnergyUnit"
 ]
 ]
}
```

## Evaluations
To run evaluations, you can use the experiments using the following command:

```
python src/saed/eval.py experiments=llm llms=azure_openai
```
Make sure to replace `experiments=llm` and `llms=azure_openai` with your specific experiment and LLM configurations as above. The output is a `results.json` file and a `results.txt` file in the `outputs` directory.

The anonymized experimental results are in the [`results`](./results) folder and the notebook to visualize the results is in the [`notebooks`](./notebooks) folder [`demo_results.ipynb`](./notebooks/demo_results.ipynb) file.
