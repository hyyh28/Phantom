from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Type

from numpy import ndarray

from model import call_api
from llm_logger import LLMLogger

logger = LLMLogger(
    log_dir="llm_agent_log",
    log_filename="llm_agent_log.txt",
    console_output=False
)


# Abstract base class for decision strategies
class DecisionStrategy(ABC):
    """
    Abstract base class for decision-making strategies.
    Different strategies can be implemented for various decision-making approaches.
    """

    @abstractmethod
    def make_decision(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Make a decision based on the provided context.
        
        Args:
            context: Dictionary containing relevant information for decision making
            
        Returns:
            numpy.ndarray: The action to take
        """
        pass


class LLMDecisionStrategy(DecisionStrategy):
    """
    Decision strategy that uses LLM to make decisions.
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the LLM decision strategy.
        
        Args:
            model: The LLM model to use
        """
        self.model = model

    def make_decision(self, context: Dict[str, Any]) -> float:
        """
        Use LLM to make a decision based on the provided context.
        
        Args:
            context: Dictionary containing agent state, environment state, etc.
            
        Returns:
            numpy.ndarray: The action to take
        """
        # Extract information from context
        env_description = context.get("env_description")
        agent_description = context.get("agent_description")
        observation = context.get('observation')
        agent_state = context.get('agent_state')
        step = context.get('step')
        history = context.get('history', [])
        agent_profile = context.get('agent_profile', {})
        action_space = context.get('action_space')

        # Format system prompt and message
        system_prompt = self._generate_system_prompt(agent_profile, action_space)
        message = self._format_prompt(env_description, agent_description, observation, agent_state, step, history, agent_profile)

        # log the prompt

        try:
            # Call LLM API
            response = call_api(self.model, message, system_prompt)

            # Parse and validate action
            action = self._parse_and_validate_action(response, action_space)
            # log the action and input
            logger.log_step(step, message, action=action)
            return action

        except Exception as e:
            print(f"LLM decision failed: {e}. Using fallback strategy.")
            return self._fallback_strategy(context)

    def _generate_system_prompt(self, agent_profile: Dict[str, Any], action_space: Any) -> str:
        """Generate a system prompt based on agent profile and action space"""
        # Customize prompt based on agent role
        role = agent_profile.get('role', 'supply chain manager')
        personality = agent_profile.get('personality', 'rational')

        prompt = f"""
        You are an AI {role} with a {personality} approach to decision making.
        Your task is to decide on the best action to take in the current situation.
        
        Action constraints:
        - Your action should be a single number
        - The valid range is from {action_space.low[0]} to {action_space.high[0]}
        - Only respond with the number, no explanation or additional text
        """

        # Add specific instructions based on agent role
        if role == 'supply chain manager':
            prompt += """
            You are managing inventory for a shop in a supply chain environment.
            Your goal is to maintain optimal inventory levels:
            - Too little inventory leads to missed sales
            - Too much inventory incurs holding costs
            - The reward is calculated as: sales - 0.1 * inventory
            """

        return prompt

    def _format_prompt(self,
                       env_description: str,
                       agent_description: str,
                       observation: np.ndarray,
                       agent_state: Any,
                       step: int,
                       history: List[Dict[str, Any]],
                       agent_profile: Dict[str, Any]) -> str:
        """Format the prompt to send to the LLM"""
        # Extract normalized values from observation
        if observation is not None:
            normalized_stock = observation[0]
            normalized_sales = observation[1]
            normalized_missed_sales = observation[2]

            # Convert to actual values based on environment constants
            max_customers = agent_profile.get('environment_constants', {}).get('num_customers', 5)
            max_order_size = agent_profile.get('environment_constants', {}).get('max_order_size', 5)
            max_stock = agent_profile.get('environment_constants', {}).get('max_stock', 100)

            max_sales_per_step = max_customers * max_order_size
            stock = int(normalized_stock * max_stock)
            sales = int(normalized_sales * max_sales_per_step)
            missed_sales = int(normalized_missed_sales * max_sales_per_step)
            prompt = env_description + "\n\n" + agent_description + "\n\n"
            prompt += f"""
            Current step: {step}
            Current inventory: {stock}/{max_stock}
            Sales this step: {sales}
            Missed sales this step: {missed_sales}
            
            History (last {len(history)} steps):
            """

            # Add history data
            for i, h in enumerate(history):
                prompt += f"Step {h['step']}: inventory={h['stock']}, sales={h['sales']}, missed sales={h['missed_sales']}\n"

            prompt += "\nBased on this information, what inventory amount should be requested from the factory?"
            return prompt

        return "No observation available. Please provide a number between 0 and 100 for inventory request."

    def _parse_and_validate_action(self, response: str, action_space: Any) -> float:
        """Parse the LLM response and validate it against the action space"""
        try:
            # Try to extract just the number from the response
            # Strip whitespace and find the first sequence of digits
            import re
            number_match = re.search(r'\d+', response.strip())
            if number_match:
                action_value = int(number_match.group())
            else:
                action_value = int(response.strip())

            # Clip to valid range
            action_value = max(action_space.low[0], min(action_value, action_space.high[0]))

            return action_value
        except:
            raise ValueError(f"Could not parse a valid number from LLM response: '{response}'")

    def _fallback_strategy(self, context: Dict[str, Any]) -> np.ndarray:
        """Fallback strategy when LLM fails"""
        action_space = context.get('action_space')
        observation = context.get('observation')

        if observation is not None:
            # Simple heuristic based on current stock
            current_stock_level = observation[0]
            max_stock = action_space.high[0]

            if current_stock_level < 0.3:  # Stock is low
                return np.array([max_stock * 0.7], dtype=np.float32)
            elif current_stock_level < 0.6:  # Stock is medium
                return np.array([max_stock * 0.4], dtype=np.float32)
            else:  # Stock is high
                return np.array([max_stock * 0.1], dtype=np.float32)

        # The default fallback is to request half of max stock
        return action_space.high[0] / 2


class HistoryManager:
    """
    Manages the history of observations and actions for an agent.
    """

    def __init__(self, max_history_length: int = 10):
        """
        Initialize the history manager.

        Args:
            max_history_length: Maximum number of history entries to keep
        """
        self.history = []
        self.max_history_length = max_history_length

    def add_entry(self, observation: np.ndarray, agent_state: Any, step: int,
                  environment_constants: Dict[str, Any]) -> None:
        """
        Add a new entry to the history.

        Args:
            observation: The observation from the environment
            agent_state: The agent's state
            step: The current step number
            environment_constants: Constants about the environment
        """
        # Extract information from observation and environment constants
        max_customers = environment_constants.get('num_customers', 5)
        max_order_size = environment_constants.get('max_order_size', 5)
        max_stock = environment_constants.get('max_stock', 100)

        max_sales_per_step = max_customers * max_order_size
        normalized_stock = observation[0]
        normalized_sales = observation[1]
        normalized_missed_sales = observation[2]

        stock = int(normalized_stock * max_stock)
        sales = int(normalized_sales * max_sales_per_step)
        missed_sales = int(normalized_missed_sales * max_sales_per_step)

        # Add entry to history
        self.history.append({
            "step": step,
            "stock": stock,
            "sales": sales,
            "missed_sales": missed_sales,
            "observation": observation.tolist()
        })

        # Maintain history size
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Get the current history"""
        return self.history

    def clear(self) -> None:
        """Clear the history"""
        self.history = []


class AgentProfile:
    """
    Defines the characteristics and personality of an agent.
    """

    def __init__(self,
                 name: str = "SHOP",
                 role: str = "supply chain manager",
                 personality: str = "rational",
                 environment_constants: Dict[str, Any] = None):
        """
        Initialize an agent profile.

        Args:
            role: The role of the agent
            personality: The personality traits of the agent
            environment_constants: Constants about the environment the agent operates in
        """
        self.name = name
        self.role = role
        self.personality = personality

        # Default environment constants for supply chain
        default_constants = {
            "num_customers": 5,
            "max_order_size": 5,
            "max_stock": 100
        }

        self.environment_constants = environment_constants if environment_constants else default_constants

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            "name": self.name,
            "role": self.role,
            "personality": self.personality,
            "environment_constants": self.environment_constants
        }


class AgentController:
    """
    Controls an agent using a specified decision strategy.
    This class decouples the agent implementation from the decision-making process.
    """

    def __init__(self,
                 decision_strategy: DecisionStrategy,
                 agent_profile: Optional[AgentProfile] = None,
                 history_manager: Optional[HistoryManager] = None,
                 env_description=None,
                 agent_description=None):
        """
        Initialize the agent controller.
        
        Args:
            decision_strategy: The strategy to use for decision making
            agent_profile: The profile of the agent
            history_manager: The manager for the agent's history
        """
        self.decision_strategy = decision_strategy
        self.agent_profile = agent_profile if agent_profile else AgentProfile()
        self.history_manager = history_manager if history_manager else HistoryManager()
        self.env_description = env_description
        self.agent_description = agent_description

    def get_action(self, observation: np.ndarray, agent_state: Any,
                   step: int, action_space: Any) -> list[ndarray]:
        """
        Get an action from the decision strategy.
        
        Args:
            observation: The observation from the environment
            agent_state: The agent's state
            step: The current step
            action_space: The action space of the environment
            
        Returns:
            numpy.ndarray: The action to take
        """
        # Update history
        self.history_manager.add_entry(
            observation,
            agent_state,
            step,
            self.agent_profile.environment_constants
        )

        # Prepare context for decision strategy
        context = {
            'env_description': self.env_description,
            'agent_description': self.agent_description,
            'observation': observation,
            'agent_state': agent_state,
            'step': step,
            'history': self.history_manager.get_history(),
            'agent_profile': self.agent_profile.to_dict(),
            'action_space': action_space
        }

        # Get action from decision strategy
        action = self.decision_strategy.make_decision(context)

        return [action]

    def reset(self) -> None:
        """Reset the agent controller"""
        self.history_manager.clear()


# Factory class for creating different types of agent controllers
class AgentControllerFactory:
    """
    Factory for creating agent controllers with different configurations.
    """

    @staticmethod
    def create_llm_agent(model: str = "gpt-4o",
                         name: str = "SHOP",
                         role: str = "ShopAgent",
                         personality: str = "rational",
                         environment_constants: Dict[str, Any] = None,
                         history_length: int = 10,
                         env_description=None,
                         agent_description=None) -> AgentController:
        """
        Create an agent controller that uses LLM for decision making.
        
        Args:
            agent_description: the description of the agent
            env_description: the description of the environment
            model: The LLM model to use
            name: The name of the agent
            role: The role of the agent
            personality: The personality of the agent
            environment_constants: Constants about the environment
            history_length: Length of history to maintain
            
        Returns:
            AgentController: An agent controller with LLM decision strategy
        """
        decision_strategy = LLMDecisionStrategy(model=model)
        agent_profile = AgentProfile(
            name=name,
            role=role,
            personality=personality,
            environment_constants=environment_constants
        )
        history_manager = HistoryManager(max_history_length=history_length)

        return AgentController(
            decision_strategy=decision_strategy,
            agent_profile=agent_profile,
            history_manager=history_manager,
            env_description=env_description,
            agent_description=agent_description
        )

    @staticmethod
    def create_custom_agent(decision_strategy: DecisionStrategy,
                            agent_profile: Optional[AgentProfile] = None,
                            history_manager: Optional[HistoryManager] = None) -> AgentController:
        """
        Create an agent controller with custom components.
        
        Args:
            decision_strategy: Custom decision strategy
            agent_profile: Custom agent profile
            history_manager: Custom history manager
            
        Returns:
            AgentController: A custom agent controller
        """
        return AgentController(
            decision_strategy=decision_strategy,
            agent_profile=agent_profile,
            history_manager=history_manager
        )


# Example usage function
def run_llm_agent_simulation(env_class, model="gpt-4o", num_episodes=1, num_steps=100,
                             agents_role=None, agents_name=None,
                             agents_personality=None):
    """
    Run a simulation using an LLM agent.
    
    Args:
        agents_name: The name of the agents to use
        env_class: The environment class to use
        model: The LLM model to use
        num_episodes: Number of episodes to run
        num_steps: Number of steps per episode
        agents_role: Role of the agents
        agents_personality: agents_personality: Personality of the agents
    """
    # Create environment
    if agents_personality is None:
        agents_personality = ["rational"]
    if agents_role is None:
        agents_role = ['ShopAgent']
    if agents_name is None:
        agents_name = ['SHOP']
    env = env_class(config_file="example_env.json")

    # Set up environment constants for the agent
    environment_constants = {
        "num_customers": 5,  # Adjust based on your environment
        "max_order_size": 5,
        "max_stock": 100
    }

    # Create agent controller
    agents_controllers = []
    for agent in agents_name:
        agent_controller = AgentControllerFactory.create_llm_agent(
            model=model,
            role=agents_role[agents_name.index(agent)],
            personality=agents_personality[agents_name.index(agent)],
            environment_constants=environment_constants,
            env_description=env.env_description,
            agent_description=env.agents[agent].agent_description
        )
        agents_controllers.append(agent_controller)

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}/{num_episodes}")

        # Reset environment and agent
        state, _ = env.reset()
        total_reward = {}
        for agent_controller in agents_controllers:
            agent_controller.reset()
            total_reward[agent_controller.agent_profile.name] = 0

        # Get action from agent_controller
        for step in range(num_steps):
            actions = {}
            for agent in agents_name:
                actions[agent] = agents_controllers[agents_name.index(agent)].get_action(
                    observation=state[agent],
                    agent_state=env.agents[agent],
                    step=step,
                    action_space=env.agents[agent].action_space
                )
            # Execute action in the environment
            state, reward, done, info, _ = env.step(actions)
            for agent in agents_name:
                total_reward[agent] += reward[agent]

            print(
                f"Step {step + 1}/{num_steps} - Action: {actions}, Reward: {reward}, Cumulative Reward: {total_reward}")

            # if done:
            #    break

        print(f"Episode {episode + 1} completed. Total reward: {total_reward}")

        # Print final state of shop agent
        shop_agent = env.agents['SHOP']
        print(f"Final inventory: {shop_agent.stock}")
        print(f"Last step sales: {shop_agent.sales}")
        print(f"Last step missed sales: {shop_agent.missed_sales}")


# Usage example
if __name__ == "__main__":
    from example_env import SupplyChainEnv

    agent_roles = ['ShopAgent']
    agent_names = ['SHOP']
    agents_personality = ['Self-Interested']
    # Run simulation with default parameters
    run_llm_agent_simulation(
        env_class=SupplyChainEnv,
        model="deepseek",
        num_episodes=1,
        num_steps=50,
        agents_personality=agents_personality
    )
    logger.save_summary()
