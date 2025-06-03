# LLM Agent Framework
## Overview
The LLM Agent Framework enables the development of intelligent agents powered by Large Language Models (LLMs). This framework allows you to create decision-making agents that operate in simulated environments such as supply chain management. The framework integrates with various LLM providers including OpenAI (GPT models), Anthropic (Claude models), and DeepSeek.
## Key Components
### 1. Model Interface () `model.py`
Provides unified interfaces to various LLM APIs:
- **Key Functions**:
    - : Universal API call interface for different models `call_api(model, message, system_prompt)`
    - : Simplified wrapper for specific models `close_source_call(model, message, system_prompt)`

- **Supported Models**:
    - Claude (Sonnet, Opus)
    - GPT-4 variants (GPT-4o, GPT-4o-mini)
    - DeepSeek models

### 2. LLM Agent Controller () `llm_control_agent.py`
The core component providing LLM-based decision mechanisms:
- **Key Classes**:
    - : Abstract base class for decision strategies `DecisionStrategy`
    - : Implementation using LLMs for decision-making `LLMDecisionStrategy`
    - `HistoryManager`: Manages agent's history of observations and actions
    - : Defines agent characteristics and personality `AgentProfile`
    - : Controls agents using specified decision strategies `AgentController`
    - : Creates different types of agent controllers `AgentControllerFactory`

- **Main APIs**:
    - `AgentControllerFactory.create_llm_agent()`: Creates an LLM-powered agent
    - `run_llm_agent_simulation()`: Runs a simulation with LLM agents

### 3. Environment Implementation
Sample environment implementation for supply chain management:
- `SupplyChainEnv`: Supply chain environment with shops, customers, and factories
- Various agent types: `ShopAgent`, `FactoryAgent`, `CustomerAgent`

### 4. Utilities and Logging
- : Provides utility functions for environment description formatting `utils.py`
- : Specialized logger for tracking LLM agent behavior `llm_logger.py`

## Customization Guide
### Creating Custom Environments
To create a new environment:
1. Create a class inheriting from `ph.PhantomEnv`
2. Define agents for your environment (inheriting from or ) `ph.Agent``ph.StrategicAgent`
3. Create a network connecting these agents
4. Provide an environment description

Example:
``` python
class MyCustomEnv(ph.PhantomEnv):
    def __init__(self, config_file=None):
        # Define your agents
        agent1 = MyCustomAgent("AGENT1")
        agent2 = MyCustomAgent("AGENT2")
        
        # Create network
        network = ph.Network([agent1, agent2])
        network.add_connection("AGENT1", "AGENT2")
        
        super().__init__(num_steps=100, network=network)
        
        # Add environment description
        self.env_description = self._build_description
```
### Creating Custom Decision Strategies
To implement a new decision strategy:
1. Create a class inheriting from `DecisionStrategy`
2. Implement the method `make_decision`

Example:
``` python
class MyCustomStrategy(DecisionStrategy):
    def make_decision(self, context: Dict[str, Any]) -> np.ndarray:
        # Implement your decision logic
        observation = context.get('observation')
        # Make a decision based on observation
        return np.array([your_decision])
```
### Customizing Agent Personality
Customize agent personality by configuring parameters in the : `AgentProfile`
``` python
agent_controller = AgentControllerFactory.create_llm_agent(
    model="gpt-4o",
    role="inventory_manager",  # Custom role
    personality="aggressive",  # Custom personality
    environment_constants={    # Custom environment constants
        "num_customers": 10,
        "max_order_size": 8,
        "max_stock": 200
    }
)
```
## Usage Example
Here's how to run a simple simulation:
``` python
from example_env import SupplyChainEnv
from llm_control_agent import run_llm_agent_simulation

run_llm_agent_simulation(
    env_class=SupplyChainEnv,
    model="gpt-4o",  # Options: "claude", "deepseek", etc.
    num_episodes=1,
    num_steps=100,
    agents_role=["ShopAgent"],
    agents_name=["SHOP"],
    agents_personality=["rational"]  # Options: "aggressive", "conservative", etc.
)
```

## Current Status and Future Development

### Completed Features

- Basic framework architecture for LLM-based agents
- Integration with multiple LLM providers (OpenAI, Anthropic, DeepSeek)
- Simple environment implementation (Supply Chain)
- Basic decision-making mechanisms
- Agent personality configuration
- Environment description system

### Pending Development

1. Agent Interactions
    - Direct communication between agents
    - Multi-agent collaboration
    - Complex negotiation protocols

2. Framework Enhancement
    - Modular architecture refactoring
    - Standardized interfaces
    - Enhanced decision strategies

3. Advanced Features
    - Complex environment implementations
    - Advanced reward mechanisms
    - Long-term memory and learning
