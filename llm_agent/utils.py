from typing import Dict, Any


def format_environment_description(config: Dict[str, Any]) -> str:
    """
    Format an environment description from a configuration dictionary

    Args:
        config: Dictionary containing environment configuration

    Returns:
        str: Formatted environment description
    """
    description = f"{config['name']}\n"
    description += "=" * len(config['name']) + "\n\n"

    # General description
    if "description" in config:
        description += f"{config['description']}\n\n"

    # Parameters section
    if "parameters" in config:
        description += "Parameters\n----------\n"
        for param_name, param_value in config["parameters"].items():
            # Format parameter name for better readability
            formatted_name = param_name.replace("_", " ").title()
            description += f"- {formatted_name}: {param_value}\n"
        description += "\n"

    # Agents section
    if "agents" in config:
        description += "Agents\n------\n"
        for agent in config["agents"]:
            agent_type = agent["type"]
            count = agent.get("count", 1)
            agent_desc = agent.get("description", "")

            # Format agent header based on count
            if count == 1:
                description += f"**{agent_type}** ({agent['id']})\n"
            else:
                description += f"**{agent_type}s** ({agent['id']}, Count: {count})\n"

            description += f"  {agent_desc}\n"

            # Add agent constraints if available
            if "constraints" in agent:
                description += "  Constraints:\n"
                for constraint in agent["constraints"]:
                    description += f"  - {constraint}\n"

            # Add agent actions if available
            if "actions" in agent:
                description += "  Actions:\n"
                for action in agent["actions"]:
                    description += f"  - {action}\n"

            # Add reward function if available
            if "reward_function" in agent:
                description += f"  Reward: {agent['reward_function']}\n"

            description += "\n"

    # Message flow section
    if "message_flow" in config:
        description += "Message Flow\n------------\n"
        for idx, msg in enumerate(config["message_flow"], 1):
            description += f"{idx}. **{msg['message']}**: {msg['from']} → {msg['to']}\n"
            if "description" in msg:
                description += f"   {msg['description']}\n"
        description += "\n"

    # Challenges section
    if "challenges" in config:
        description += "Key Challenges\n-------------\n"
        for challenge in config["challenges"]:
            description += f"- {challenge}\n"

    return description
