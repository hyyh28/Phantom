import json
from typing import Dict, Any, List, Optional, Tuple, Union
import phantom as ph
from model import call_api, build_deepseek_grad_engine
import textgrad as tg

# Import message payloads and agent classes from agent.py
from agent import (
    OrderRequest, OrderResponse, StockRequest, StockResponse,
    FactoryAgent, CustomerAgent, LLMCustomerAgent, ShopAgent,
    DistributorAgent, ManufacturingAgent, ProducreAgent, RetailAgent, TransAgent
)

# Constants
DEFAULT_MODEL = "deepseek"
DEFAULT_STEPS = 100

# System prompt for the LLM
SYSTEM_PROMPT = """
You are a supply chain environment builder. Your task is to analyze the provided environment description 
and create a structured representation of entities, their relationships, and objectives in a supply chain 
simulation. Focus only on the environment creation aspects, not on agent implementation.
"""


class EnvironmentBuilder:
    """
    LLM-based environment builder that constructs Phantom environments based on user descriptions.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.available_agents = {
            "Factory": FactoryAgent,
            "Customer": CustomerAgent,
            "LLMCustomer": LLMCustomerAgent,
            "Shop": ShopAgent,
            "Distributor": DistributorAgent,
            "Manufacturing": ManufacturingAgent,
            "Producer": ProducreAgent,
            "Retail": RetailAgent,
            "Transport": TransAgent
        }

    def parse_description(self, description: str) -> Dict[str, Any]:
        """
        Use LLM to parse the environment description into a structured format

        Args:
            description: The text description of the environment

        Returns:
            A dictionary containing the structured environment specification
        """
        prompt = f"""
Please analyze the following supply chain environment description and extract:
1. Entities with their roles and whether they are learning agents
2. Relationships between entities (interactions, flows of goods/information)
3. The learning objective for the system

Please format your response as a JSON object with the following structure:
{{
  "entities": [
    {{
      "name": "entity_name",
      "type": "learning" or "non-learning",
      "role": "brief description of role",
      "agent_class": "appropriate agent class from the available options"
    }},
    ...
  ],
  "relationships": [
    {{
      "source": "source_entity_name",
      "target": "target_entity_name",
      "interaction_type": "type of interaction (e.g. 'order', 'supply')",
      "description": "description of the relationship"
    }},
    ...
  ],
  "objective": {{
    "learning_entity": "name of the learning entity",
    "decision": "what the entity is learning to decide",
    "goal": "optimization objective"
  }}
}}

Available agent classes are: Factory, Customer, LLMCustomer, Shop, Distributor, Manufacturing, Producer, Retail, Transport.
Choose the most appropriate class for each entity based on its role in the supply chain.

Environment description:
{description}

Return only the JSON object without any additional text.
"""
        # initilize textgrad engine
        engine = build_deepseek_grad_engine()
        tg.set_backward_engine(engine=engine, override=True)
        model = tg.BlackboxLLM(engine)

        # input question
        question = tg.Variable(prompt, role_description=SYSTEM_PROMPT, requires_grad=False)

        # initilized results x_0
        response = model(question)
        response.set_role_description("JSON structured environment specification")
        response.requires_grad = True

        # vertify the results in Json
        try:
            start_idx = response.value.find('{')
            end_idx = response.value.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("Could not find valid JSON in LLM response")

            json_str = response.value[start_idx:end_idx + 1]
            parsed_json = json.loads(json_str)
            params = self.generate_environment_init_params(parsed_json)
            errors_list = check_params(params)

            # use textgrad to update the results
            if errors_list:
                errors = "\n".join(errors_list)
                print(errors)
                # optimizer
                optimizer = tg.TGD(parameters=[response])

                # create evaluation_instruction
                evaluation_instruction = tg.Variable(
                    f"Here's the environment description: {description}\n\n"
                    f"The current JSON response has the following errors:\n{errors}\n\n"
                    f"Fix these errors in the JSON structure without changing its format. "
                    f"Make sure all agent classes are valid and relationships are properly defined.",
                    role_description=SYSTEM_PROMPT,
                    requires_grad=False
                )

                # create loss function
                loss_fn = tg.TextLoss(evaluation_instruction)

                # optimization loop
                for _ in range(3):
                    loss = loss_fn(response)
                    loss.backward()
                    optimizer.step()

                    # recheck errors
                    try:
                        start_idx = response.value.find('{')
                        end_idx = response.value.rfind('}')
                        if start_idx != -1 and end_idx != -1:
                            json_str = response.value[start_idx:end_idx + 1]
                            parsed_json = json.loads(json_str)
                            params = self.generate_environment_init_params(parsed_json)
                            new_errors = check_params(params)
                            if not new_errors:
                                break
                    except:
                        pass

            # get the final results
            start_idx = response.value.find('{')
            end_idx = response.value.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("Could not find valid JSON in optimized LLM response")

            json_str = response.value[start_idx:end_idx + 1]
            return json.loads(json_str)

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")

    def generate_environment_init_params(self, env_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate initialization parameters for the environment based on the specification

        Args:
            env_spec: The environment specification

        Returns:
            Dictionary of parameters for environment initialization
        """
        # Extract entities and relationships
        entities = env_spec.get("entities", [])
        relationships = env_spec.get("relationships", [])
        objective = env_spec.get("objective", {})

        # Setup params dictionary
        params = {
            "entities": entities,
            "relationships": relationships,
            "objective": objective,
        }

        return params

    def build_environment(self, description: str) -> "SupplyChainEnv":
        """
        Build a complete environment from a text description

        Args:
            description: The text description of the environment

        Returns:
            An initialized SupplyChainEnv instance
        """
        # Parse the description
        env_spec = self.parse_description(description)

        # Generate environment parameters
        params = self.generate_environment_init_params(env_spec)
        # check(params)

        # Create and return the environment
        return SupplyChainEnv(params)


def check_params(params: Dict[str, Any] = None):
    errors = []
    params = params
    entities = params.get("entities", [])
    relationships = params.get("relationships", [])

    # Agent class mapping
    agent_classes = {
        "Factory": FactoryAgent,
        "Customer": CustomerAgent,
        "LLMCustomer": LLMCustomerAgent,
        "Shop": ShopAgent,
        "Distributor": DistributorAgent,
        "Manufacturing": ManufacturingAgent,
        "Producer": ProducreAgent,
        "Retail": RetailAgent,
        "Transport": TransAgent
    }
    # Create all agents
    for entity in entities:
        name = entity["name"]
        agent_class_name = entity.get("agent_class", "")
        if agent_class_name not in agent_classes:
            errors.append(f"Unknown agent class: {agent_class_name}")
        elif agent_class_name == "Customer" or agent_class_name == "LLMCustomer":
            shop_connections = [r for r in relationships
                                if r["source"] == name and "shop" in r["target"].lower()]
            shop_id = shop_connections[0]["target"] if shop_connections else None
            if not shop_id:
                errors.append(f"Customer {name} has no shop connection defined")
        elif agent_class_name == "Shop":
            factory_connections = [r for r in relationships
                                   if r["source"] == name and "warehouse" in r["target"].lower()]
            factory_id = factory_connections[0]["target"] if factory_connections else None
            if not factory_id:
                errors.append(f"Shop {name} has no factory connection defined")
    return errors


class SupplyChainEnv(ph.PhantomEnv):
    """
    A Phantom environment for supply chain simulations built from a specification.
    """

    def __init__(self, config: Dict[str, Any], num_steps: int = DEFAULT_STEPS, entities=None, relationships=None, objective=None):
        """
        Initialize the supply chain environment

        Args:
            config: The environment configuration containing entities, relationships, and objective
            num_steps: The number of steps per episode
        """
        self.config = config
        self.entities = config.get("entities", [])
        self.relationships = config.get("relationships", [])
        self.objective = config.get("objective", {})

        # Create agents based on specifications
        self.agent_instances = {}

        # Agent class mapping
        agent_classes = {
            "Factory": FactoryAgent,
            "Customer": CustomerAgent,
            "LLMCustomer": LLMCustomerAgent,
            "Shop": ShopAgent,
            "Distributor": DistributorAgent,
            "Manufacturing": ManufacturingAgent,
            "Producer": ProducreAgent,
            "Retail": RetailAgent,
            "Transport": TransAgent
        }

        # Create all agents
        for entity in self.entities:
            name = entity["name"]
            agent_class_name = entity.get("agent_class", "")

            if agent_class_name not in agent_classes:
                raise ValueError(f"Unknown agent class: {agent_class_name}")

            # Special handling for different agent types
            if agent_class_name == "Factory":
                self.agent_instances[name] = FactoryAgent(name)
            elif agent_class_name == "Customer":
                # Find shop entity to connect to
                shop_connections = [r for r in self.relationships
                                    if r["source"] == name and "shop" in r["target"].lower()]
                shop_id = shop_connections[0]["target"] if shop_connections else None

                if not shop_id:
                    raise ValueError(f"Customer {name} has no shop connection defined")

                self.agent_instances[name] = CustomerAgent(name, shop_id)
            elif agent_class_name == "LLMCustomer":
                # Find shop connection
                shop_connections = [r for r in self.relationships
                                    if r["source"] == name and "shop" in r["target"].lower()]
                shop_id = shop_connections[0]["target"] if shop_connections else None

                if not shop_id:
                    raise ValueError(f"LLMCustomer {name} has no shop connection defined")

                # Create with random personality
                personality = entity.get("role", "a regular customer")
                self.agent_instances[name] = LLMCustomerAgent(name, shop_id, personality)
            elif agent_class_name == "Shop":
                # Find factory connection
                factory_connections = [r for r in self.relationships
                                       if r["source"] == name and "warehouse" in r["target"].lower()]
                factory_id = factory_connections[0]["target"] if factory_connections else None

                if not factory_id:
                    raise ValueError(f"Shop {name} has no factory connection defined")

                self.agent_instances[name] = ShopAgent(name, factory_id)
            else:
                # Generic initialization for other agent types
                self.agent_instances[name] = agent_classes[agent_class_name](name)

        # Create network with all agents
        self.network = ph.Network(list(self.agent_instances.values()))

        # Add connections based on relationships
        for relationship in self.relationships:
            source = relationship["source"]
            target = relationship["target"]

            # Ensure both source and target exist
            if source in self.agent_instances and target in self.agent_instances:
                self.network.add_connection(source, target)

        # Initialize the base class
        super().__init__(num_steps=num_steps, network=self.network)

        # Store environment description for reference
        self.description = self._build_description()

    def _build_description(self) -> str:
        """
        Build a human-readable description of the environment

        Returns:
            A string describing the environment
        """
        lines = ["Supply Chain Environment Description:"]

        lines.append("\n1. Entities:")
        for entity in self.entities:
            entity_type = "Learning Agent" if entity["type"] == "learning" else "Non-learning Agent"
            lines.append(f"  - {entity['name']} ({entity_type}): {entity.get('role', '')}")

        lines.append("\n2. Relationships:")
        for rel in self.relationships:
            lines.append(f"  - {rel['source']} → {rel['target']}: {rel.get('description', '')}")

        lines.append("\n3. Learning Objective:")
        if self.objective:
            lines.append(f"  - Learning Entity: {self.objective.get('learning_entity', '')}")
            lines.append(f"  - Decision: {self.objective.get('decision', '')}")
            lines.append(f"  - Goal: {self.objective.get('goal', '')}")

        return "\n".join(lines)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        # Reset all agents
        for agent in self.agent_instances.values():
            agent.reset()

        return super().reset()


def build_environment_from_description(description: str, model: str = DEFAULT_MODEL) -> SupplyChainEnv:
    """
    Main function to build an environment from a text description

    Args:
        description: The environment description
        model: The LLM model to use

    Returns:
        An initialized environment
    """
    builder = EnvironmentBuilder(model)
    return builder.build_environment(description)


if __name__ == "__main__":
    # Example usage
    description = """
1. Entities
* 1 Factory(NAME: WAREHOUSE) (non-learning) – Produces goods upon request from the shop.
* 1 Shop(Name: SHOP) (learning) – Maintains inventory and decides how much to order from the factory.
* 5 Customers(Name: CUST_1 to CUST_5) (non-learning) – Generate demand stochastically each time step.

2. Relationships Between Entities
* Shop places orders to Factory, which delivers after a fixed lead time.
* Customers interact with any shop to purchase products.
* Unmet demand may result in backlogs or lost sales.
* Shop updates its inventory policy via learning.

3. Objective
The goal is for the Shop to learn an inventory policy that minimizes total costs (including holding, shortage, and ordering costs) while meeting stochastic customer demand efficiently.
    """

    env = build_environment_from_description(description)
    print(env.description)
