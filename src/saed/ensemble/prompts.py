from langchain_core.prompts import ChatPromptTemplate

edm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant who helps map tabular data columns to ontology classes. You have expert knowledge in semantic table annotation, i.e., table column-to-ontology class mappings. Your goal is to determine the most semantically appropriate ontology class (or set of classes) for a given column, based on the provided table name, table header, example data, column name, and the candidate ontology classes."
        ),
        (
            "human",
            """
We have a table named '{table_name}' which has several columns. Below is the table in Markdown format, including column headers and a few example rows:

{table_in_markdown}

We are currently focusing on ontology classes at the given level. Below are a subset of the ontology classes available at this level:
{current_level_ontology_classes}

We want to determine the best fitting ontology class (or class path if multiple levels are considered) for the following column: '{column_name}'.

Instructions:
1. Review the column name and the example data.
2. Based on the ontology classes provided, select the most suitable ontology class or classes (or ancestor class or classes) of the corresponding class that best describe the semantic meaning of this column.
3. Output the final answer enclosed in <answer></answer> tags, 
    3.1 If multiple ontology classes are required to describe the column, split the classes with a comma, for example, <answer>ontology_class1, ontology_class2, ..., ontology_classN</answer>
    3.2 If no suitable class is found, respond with <answer>-</answer>.
Note: The provided set of ontology classes might not be complete. If you think none of the given classes are suitable, you can indicate that accordingly.
 """
        ),
    ]
)


llm_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant who helps map tabular data columns to ontology classes. You have expert knowledge in semantic table annotation, i.e., table column-to-ontology class mappings. Your goal is to determine the most semantically appropriate ontology class ontology class (or set of classes) for a given column, based on the provided table name, table header, example data, column name, and the ontology classes at current level."
        ),
        (
            "human",
            """
We have a table named '{table_name}' which has several columns. Below is the table in Markdown format, including column headers and a few example rows:

{table_in_markdown}

We are currently focusing on ontology classes at the given level. Below are the set of the ontology classes available at this level:
{current_level_ontology_classes}

We want to determine the best fitting ontology class (or class path if multiple levels are considered) for the following column: '{column_name}'.

Instructions:
1. Review the column name and the example data.
2. Based on the ontology classes provided, select the most suitable ontology class or classes (or ancestor class or classes) of the corresponding class that best describe the semantic meaning of this column.
3. Output the final answer enclosed in <answer></answer> tags, 
    3.1 If multiple ontology classes are required to describe the column, split the classes with a comma, for example, <answer>ontology_class1, ontology_class2, ..., ontology_classN</answer>
    3.2 If no suitable class is found, respond with <answer>-</answer>.
 """
        ),
    ]
)

cot_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a knowledgeable assistant who helps map tabular data columns to ontology classes. You have expert knowledge in semantic table annotation, i.e., table column-to-ontology class mappings. Your goal is to determine the most semantically appropriate ontology class ontology class (or set of classes) for a given column, based on the provided table name, table header, example data, column name, and the ontology classes at current level."
        ),
        (
            "human",
            """
We have a table named '{table_name}' which has several columns. Below is the table in Markdown format, including column headers and a few example rows:

{table_in_markdown}

We are currently focusing on ontology classes at the given level. Below are the set of the ontology classes available at this level:
{current_level_ontology_classes}

We want to determine the best fitting ontology class (or class path if multiple levels are considered) for the following column: '{column_name}'.

Instructions:
1. Review the column name and the example data.
2. Based on the ontology classes provided, select the most suitable ontology class or classes (or ancestor class or classes) of the corresponding class that best describe the semantic meaning of this column.
3. First, output your reasoning enclosed in <reasoning></reasoning> tags. Then output the final answer enclosed in <answer></answer> tags, 
    3.1 If multiple ontology classes are required to describe the column, split the classes with a comma, for example, <answer>ontology_class1, ontology_class2, ..., ontology_classN</answer>
    3.2 If no suitable class is found, respond with <answer>-</answer>.
 """
        ),
    ]
)