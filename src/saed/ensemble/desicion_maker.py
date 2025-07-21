import random
import re
from collections import defaultdict
from omegaconf import DictConfig

from saed.ensemble.llms import LLM

class DecisionMaker:
    """
    A class that represents a collaborative decision-making algorithm using LLM-based agents.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.llm = LLM(config)
    
    def decision_making(self, table_name, table_in_markdown, column_name, current_level_ontology_classes) -> str:
        if self.config['experiments']['mode'] == 'llm' or self.config['experiments']['mode'] == 'cot':
            print('Mode:', self.config['experiments']['mode'])
            return self.llm_decision_making(table_name, table_in_markdown, column_name, current_level_ontology_classes)
        elif self.config['experiments']['mode'] == 'edm':
            return self.ensemble_decision_making(table_name, table_in_markdown, column_name, current_level_ontology_classes)
    
    def llm_decision_making(self, table_name, table_in_markdown, column_name, current_level_ontology_classes) -> str:
        """
        A mock implementation of a LLM-based decision-making algorithm.
        
        Parameters:
            table_name (str): Name of the table.
            table_in_markdown (str): The table data in markdown format.
            column_name (str): The column name that we are mapping.
            current_level_ontology_classes (list): A list of classes (e.g. ["a", "b", "c", "d", "e"]).
        Returns:
            str: "-" if no suitable class is found, or a comma-separated string of class names that reached consensus.
        """
        data = {
            "table_name": table_name,
            "table_in_markdown": table_in_markdown,
            "column_name": column_name,
            "current_level_ontology_classes": ", ".join(current_level_ontology_classes)
        }
        result = self.llm.generate(data)
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_matches = re.findall(answer_pattern, result, flags=re.DOTALL)
        if answer_matches:
            answer_content = answer_matches[0]
            if  answer_content == "-":
                return "-"
            else:
                predicted_ontology_classes = answer_content.split(", ")
                selected_ontology_classes = []
                for predicted_ontology_class in predicted_ontology_classes:
                    if predicted_ontology_class in current_level_ontology_classes:
                        selected_ontology_classes.append(predicted_ontology_class)
                if selected_ontology_classes:
                    return ", ".join(selected_ontology_classes)
        else:
            return "-"
        
    def ensemble_decision_making(self, table_name, table_in_markdown, column_name, current_level_ontology_classes) -> str:
        """
        A mock implementation of a LLM-based collaborative decision-making algorithm.
        This function:
        - Distributes classes to agents based on given averages.
        - Each agent votes using an LLM chain call.
        - Aggregates results to form a consensus.
        
        Parameters:
            table_name (str): Name of the table.
            table_in_markdown (str): The table data in markdown format.
            column_name (str): The column name that we are mapping.
            current_level_ontology_classes (list): A list of classes (e.g. ["a", "b", "c", "d", "e"]).
            avg_classes_per_agent (int): Average number of classes per agent.
            avg_agents_per_class (int): Average number of agents that see each class.
            consensus_threshold_ratio (float): The ratio of supporting votes needed from the agents 
                                            that saw a class to consider it selected.
        Returns:
            str: "-" if no suitable class is found, or a comma-separated string of class names that reached consensus.
        """
        avg_classes_per_agent=self.config['experiments']['avg_classes_per_agent']
        avg_agents_per_class=self.config['experiments']['avg_agents_per_class']
        consensus_threshold_ratio=self.config['experiments']['consensus_threshold_ratio']
        classes = current_level_ontology_classes    
        if not classes:
            return "-"
        num_classes = len(classes)
        # Compute the approximate number of agents needed:
        # num_agents * avg_classes_per_agent ≈ num_classes * avg_agents_per_class
        # => num_agents ≈ (num_classes * avg_agents_per_class) / avg_classes_per_agent
        num_agents = max(avg_agents_per_class, (num_classes * avg_agents_per_class) // avg_classes_per_agent + 1)
        
        # Assign classes to agents
        agents_assignments = [[] for _ in range(num_agents)]
        for class_ in classes:
            if num_agents < avg_agents_per_class:
                # Assign all agents to this class if not enough agents
                assigned_agents = range(num_agents)
            else:
                assigned_agents = random.sample(range(num_agents), avg_agents_per_class)
            for agent in assigned_agents:
                agents_assignments[agent].append(class_)

        if not agents_assignments:
            # If something goes wrong, fallback to one agent seeing all classes
            agents_assignments = [classes]
        # Count how many agents see each class (for consensus calculation)
        agents_that_saw_class = defaultdict(int)
        for class_ in classes:
            agents_that_saw_class[class_] = sum(1 for agent_c in agents_assignments if class_ in agent_c)
        
        # Collect votes
        votes_per_class = defaultdict(int)
        # Query each agent
        for agent_classes in agents_assignments:
            assigned_classes_str = ", ".join(agent_classes)
            data = {
                "table_name": table_name,
                "table_in_markdown": table_in_markdown,
                "column_name": column_name,
                "current_level_ontology_classes": assigned_classes_str
            }
            result = self.llm.generate(data)
            
            answer_pattern = r"<answer>(.*?)</answer>"
            answer_matches = re.findall(answer_pattern, result, flags=re.DOTALL)
            if answer_matches:
                answer_content = answer_matches[0]
                if answer_content != "-":
                    chosen_classes = [r.strip() for r in answer_content.split(",")]
                    # Increment votes only for classes the agent actually sees
                    for cc in chosen_classes:
                        if cc in agent_classes:
                            votes_per_class[cc] += 1
        # Determine consensus
        selected_classes = []
        for class_ in classes:
            if agents_that_saw_class[class_] > 0:
                ratio = votes_per_class[class_] / agents_that_saw_class[class_]
                if ratio >= consensus_threshold_ratio and votes_per_class[class_] > 0:
                    selected_classes.append(class_)
        if not selected_classes:
            return "-"
        else:
            return ", ".join(selected_classes)