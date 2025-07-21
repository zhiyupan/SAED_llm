
from omegaconf import DictConfig
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from saed.ensemble.prompts import *

class LLM:
    """
    A class to represent the LLMs in the ensemble.
    """
    
    def __init__(self, config: DictConfig):
        """
        Constructor for the LLM class.
        
        Parameters:
            model (str): The model to use for the LLM.
            prompt (str): The prompt to use for the LLM.
        """
        self.config = config
        if self.config['llms']['name'] == "ollama":
            self.llm = OllamaLLM(base_url=self.config['llms']['base_url'], model=self.config['llms']['model'])
        elif self.config['llms']['name'] == "azure_openai":
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.config['llms']['endpoint'],
                azure_deployment=self.config['llms']['deployment_name'],  # or your deployment
                api_version=self.config['llms']['api_version'],  # or your api version
                api_key=self.config['llms']['api_key'],
                temperature=self.config['llms']['temperature'],
                max_tokens=None,
                timeout=None,
            )
        else:
            raise ValueError("Invalid LLM name")
        
        # self.prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "You are a helpful assistant that translates English to French. Translate the user sentence.",
        #         ),
        #         ("human", "I love {programming_language} programming.")
        #     ]
        # )
        
        if config['experiments']['mode'] == "edm":
            self.prompt = edm_prompt
        elif config['experiments']['mode'] == "llm":
            self.prompt = llm_prompt
        elif config['experiments']['mode'] == "cot":
            self.prompt = cot_prompt
        else:
            raise ValueError("Invalid experiment mode")
        self.chain = self.prompt | self.llm

    def generate(self, data) -> str:
        # result = self.chain.invoke(
        #     {
        #         "programming_language": data["programming_language"]
        #     }
        # )
        result = self.chain.invoke(
            {
                "table_name": data["table_name"],
                "table_in_markdown": data["table_in_markdown"],
                "column_name": data["column_name"],
                "current_level_ontology_classes": data["current_level_ontology_classes"]
            }
        )
        if self.config['llms']['name'] == "ollama":
            return result
        elif self.config['llms']['name'] == "azure_openai":
            return result.content
        