import gymnasium as gym
import numpy as np
import phantom as ph
from model import call_api

NUM_EPISODE_STEPS = 100

NUM_CUSTOMERS = 5
CUSTOMER_MAX_ORDER_SIZE = 5
SHOP_MAX_STOCK = 100


@ph.msg_payload("CustomerAgent", "ShopAgent")
class OrderRequest:
    size: int


@ph.msg_payload("ShopAgent", "CustomerAgent")
class OrderResponse:
    size: int


@ph.msg_payload("ShopAgent", "FactoryAgent")
class StockRequest:
    size: int


@ph.msg_payload("FactoryAgent", "ShopAgent")
class StockResponse:
    size: int


class FactoryAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    @ph.agents.msg_handler(StockRequest)
    def handle_stock_request(self, ctx: ph.Context, message: ph.Message):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        return [(message.sender_id, StockResponse(message.payload.size))]


class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: ph.AgentID, shop_id: ph.AgentID):
        super().__init__(agent_id)

        # We need to store the shop's ID so we know who to send order requests to.
        self.shop_id: str = shop_id

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        # The customer will receive it's order from the shop but we do not need to take
        # any actions on it.
        return

    def generate_messages(self, ctx: ph.Context):
        # At the start of each step we generate an order with a random size to send to
        # the shop.
        order_size = np.random.randint(CUSTOMER_MAX_ORDER_SIZE)

        # We perform this action by sending a stock request message to the factory.
        return [(self.shop_id, OrderRequest(order_size))]


class LLMCustomerAgent(CustomerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_id: ph.AgentID, personality: str, model: str = "sonnet"):
        super().__init__(agent_id, shop_id)
        self.personality = personality
        self.model = model
        self.agent_description = f"""You are a customer with the following personality: {personality}. 
        You need to decide how much to order from a shop that sells items. The maximum order size is {CUSTOMER_MAX_ORDER_SIZE}.
        You should make ordering decisions based on your personality traits."""

    def generate_messages(self, ctx: ph.Context):
        # Generate prompt for LLM to decide order size
        prompt = f"Based on your personality, how many items would you order? Maximum order size is {CUSTOMER_MAX_ORDER_SIZE}. Respond with just a number."

        # Get order size from LLM
        response = call_api(self.model, prompt, self.agent_description)
        try:
            order_size = min(int(response), CUSTOMER_MAX_ORDER_SIZE)
        except (ValueError, TypeError):
            order_size = np.random.randint(CUSTOMER_MAX_ORDER_SIZE)

        return [(self.shop_id, OrderRequest(order_size))]

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        # The LLM customer receives an order response but doesn't need to take action
        return


class ShopAgent(ph.StrategicAgent):
    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has...
        self.stock: int = 0

        # ...and how many sales have been made...
        self.sales: int = 0

        # ...and how many sales per step the shop has missed due to not having enough
        # stocks.
        self.missed_sales: int = 0

        # = [Stock, Sales, Missed Sales]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))

        # = [Restock Quantity]
        self.action_space = gym.spaces.Box(low=0.0, high=SHOP_MAX_STOCK, shape=(1,))
        self.agent_description = """You are the ShopAgent acts as an intermediary between the FactoryAgent and CustomerAgents in a supply chain simulation. You maintain inventory, processes customer orders, and manages stock levels by requesting resupply from the factory. You aim to maximize sales while minimizing excess inventory costs through strategic restocking decisions. You track current stock levels, completed sales, and missed sales opportunities due to stockouts."""

    def pre_message_resolution(self, ctx: ph.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the factory contain stock.
        self.delivered_stock = message.payload.size

        self.stock = min(self.stock + self.delivered_stock, SHOP_MAX_STOCK)

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(self, ctx: ph.Context, message: ph.Message):
        amount_requested = message.payload.size

        # If the order size is more than the amount of stock, partially fill the order.
        if amount_requested > self.stock:
            self.missed_sales += amount_requested - self.stock
            stock_to_sell = self.stock
            self.stock = 0
        # ... Otherwise completely fill the order.
        else:
            stock_to_sell = amount_requested
            self.stock -= amount_requested

        self.sales += stock_to_sell

        # Send the customer their order.
        return [(message.sender_id, OrderResponse(stock_to_sell))]

    def encode_observation(self, ctx: ph.Context):
        max_sales_per_step = NUM_CUSTOMERS * CUSTOMER_MAX_ORDER_SIZE

        return np.array(
            [
                self.stock / SHOP_MAX_STOCK,
                self.sales / max_sales_per_step,
                self.missed_sales / max_sales_per_step,
            ],
            dtype=np.float32,
        )

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from
        # the factory, clipped so the shop never requests more stock than it can hold.
        stock_to_request = min(int(round(action[0])), SHOP_MAX_STOCK - self.stock)

        # We perform this action by sending a stock request message to the factory.
        return [(self.factory_id, StockRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto excess stock.
        return self.sales - 0.1 * self.stock

    def reset(self):
        self.stock = 0
