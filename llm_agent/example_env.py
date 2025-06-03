import phantom as ph
import json
from typing import Optional
from utils import format_environment_description
from agent import FactoryAgent, CustomerAgent, ShopAgent, NUM_CUSTOMERS, NUM_EPISODE_STEPS


class SupplyChainEnv(ph.PhantomEnv):
    def __init__(self, config_file: Optional[str] = None):
        # Store the config file path
        self.config_file = config_file

        # Define agent IDs
        factory_id = "WAREHOUSE"
        customer_ids = [f"CUST{i + 1}" for i in range(NUM_CUSTOMERS)]
        shop_id = "SHOP"

        factory_agent = FactoryAgent(factory_id)
        customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in customer_ids]
        shop_agent = ShopAgent(shop_id, factory_id=factory_id)

        agents = [shop_agent, factory_agent] + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shop to the factory
        network.add_connection(shop_id, factory_id)

        # Connect the shop to the customers
        network.add_connections_between([shop_id], customer_ids)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network)

        # Add environment description
        self.env_description = self._build_description

    @property
    def _build_description(self) -> str:
        """
        Build a description of the environment using a configuration file if provided
        
        Returns:
            str: A detailed description of the environment
        """
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            return format_environment_description(config)


env = SupplyChainEnv(config_file="example_env.json")
env.reset()