# LLM Agent Framework

## Overview

The LLM Agent Framework enables the development of intelligent agents powered by Large Language Models (LLMs). This
framework allows you to create decision-making agents that operate in simulated environments such as supply chain
management. The framework integrates with various LLM providers including OpenAI (GPT models), Anthropic (Claude
models), and DeepSeek.

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

### 4. Utilities and Logging

- : Provides utility functions for environment description formatting `utils.py`
- : Specialized logger for tracking LLM agent behavior `llm_logger.py`

### 5. Environment Autobuilder (`env_autobuilder.py`)

The Environment Autobuilder enables automatic construction of simulation environments based on text descriptions using textgrad optimization:

- **Key Functionality**:
  - Parses natural language descriptions into structured environment specifications
  - Dynamically creates agents and relationships based on the specification
  - Builds a complete `SupplyChainEnv` with proper connections between entities

- **TextGrad Integration**:
  - Uses gradient-based optimization to refine environment structures
  - Iteratively improves JSON representations when errors are detected
  - Ensures created environments properly reflect user descriptions

- **Main Components**:
  - `EnvironmentBuilder`: Core class for parsing descriptions and building environments
  - `parse_description()`: Converts text to structured specifications using LLMs and TextGrad
  - `build_environment()`: Instantiates a complete environment from specifications
  - `SupplyChainEnv`: Environment implementation with dynamically created agents

Example usage:
```python
description = """
1. Entities
* 1 Factory (non-learning) – Produces goods upon request from the shop.
* 1 Shop (learning) – Maintains inventory and decides how much to order.
* 5 Customers (non-learning) – Generate demand stochastically.

2. Relationships Between Entities
* Shop places orders to Factory, which delivers after a fixed lead time.
* Customers interact with the shop to purchase products.

3. Objective
The goal is for the Shop to learn an inventory policy that minimizes total costs.
"""

env = build_environment_from_description(description)
```