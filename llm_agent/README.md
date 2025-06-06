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
- `AIToBuilderEnv`: Generic builder environment for creating synthetic data and scenarios
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