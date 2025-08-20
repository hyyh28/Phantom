import gymnasium as gym
import numpy as np
import phantom as ph

# ====== 配置参数 ======
NUM_EPISODE_STEPS = 100
MAX_ORDER = 20
MAX_INVENTORY = 100


# ====== 消息定义 ======
@ph.msg_payload("DownstreamAgent", "UpstreamAgent")
class OrderRequest:
    size: int


@ph.msg_payload("UpstreamAgent", "DownstreamAgent")
class Delivery:
    size: int


# ====== 代理定义 ======
class SupplyChainAgent(ph.StrategicAgent):
    def __init__(self, agent_id: str, upstream_id: str | None, downstream_id: str | None, c_h=0.5, c_p=1.0):
        super().__init__(agent_id)
        self.upstream_id = upstream_id
        self.downstream_id = downstream_id

        # 状态
        self.stock = 10   # 初始库存
        self.backlog = 0  # 未满足订单
        self.sales = 0
        self.missed = 0

        # 成本参数
        self.c_h = c_h
        self.c_p = c_p

        # Gym空间
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(MAX_ORDER + 1)

    def pre_message_resolution(self, ctx: ph.Context):
        self.sales = 0
        self.missed = 0

    @ph.agents.msg_handler(OrderRequest)
    def handle_order(self, ctx: ph.Context, message: ph.Message):
        demand = message.payload.size
        # 更新 backlog
        self.backlog += demand

        # 满足订单
        fulfilled = min(self.stock, self.backlog)
        self.stock -= fulfilled
        self.backlog -= fulfilled
        self.sales += fulfilled
        self.missed = self.backlog

        # 向下游发货
        return [(message.sender_id, Delivery(fulfilled))]

    @ph.agents.msg_handler(Delivery)
    def handle_delivery(self, ctx: ph.Context, message: ph.Message):
        # 上游发货到货
        self.stock = min(self.stock + message.payload.size, MAX_INVENTORY)

    def encode_observation(self, ctx: ph.Context):
        return np.array(
            [
                self.stock / MAX_INVENTORY,
                self.sales / MAX_ORDER,
                self.backlog / MAX_ORDER,
            ],
            dtype=np.float32,
        )

    def decode_action(self, ctx: ph.Context, action: int):
        if self.upstream_id is None:
            return []
        order_qty = int(action)
        return [(self.upstream_id, OrderRequest(order_qty))]

    def compute_reward(self, ctx: ph.Context) -> float:
        # 成本 = 积压成本 + 库存成本
        cost = self.c_p * self.backlog + self.c_h * self.stock
        return -cost

    def reset(self):
        self.stock = 10
        self.backlog = 0


# ====== 顶层环境 ======
class BeerGameEnv(ph.PhantomEnv):
    def __init__(self):
        # 定义角色ID
        retailer_id = "RETAILER"
        wholesaler_id = "WHOLESALER"
        distributor_id = "DISTRIBUTOR"
        factory_id = "FACTORY"

        # 定义代理
        retailer = SupplyChainAgent(retailer_id, wholesaler_id, None)
        wholesaler = SupplyChainAgent(wholesaler_id, distributor_id, retailer_id)
        distributor = SupplyChainAgent(distributor_id, factory_id, wholesaler_id)
        factory = SupplyChainAgent(factory_id, None, distributor_id)

        agents = [retailer, wholesaler, distributor, factory]
        network = ph.Network(agents)

        # 建立连接
        network.add_connection(retailer_id, wholesaler_id)
        network.add_connection(wholesaler_id, distributor_id)
        network.add_connection(distributor_id, factory_id)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network)
