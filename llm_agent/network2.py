import gymnasium as gym
import numpy as np
import phantom as ph

# ====== 配置参数 ======
NUM_EPISODE_STEPS = 100
MAX_ORDER = 20
MAX_INVENTORY = 100


# ====== 消息定义 ======
# @ph.msg_payload("SupplyChainAgent", "SupplyChainAgent")
# class OrderRequest:
#     size: int


# @ph.msg_payload("SupplyChainAgent", "SupplyChainAgent")
# class Delivery:
#     size: int

@ph.msg_payload("DownstreamAgent", "UpstreamAgent")
class OrderRequest:
    size: int


@ph.msg_payload("UpstreamAgent", "DownstreamAgent")
class Delivery:
    size: int

@ph.msg_payload("CustomerAgent", "SupplyChainAgent")
class OrderRequest1:
    size: int

@ph.msg_payload("SupplyChainAgent", "CustomerAgent")
class Delivery1:
    size: int


# ====== 代理定义 ======
class SupplyChainAgent(ph.StrategicAgent):
    def __init__(self, agent_id: str, upstream_id: str | None, downstream_id: str | None, c_h=0.5, c_p=1.0):
        super().__init__(agent_id)
        self.upstream_id = upstream_id
        self.downstream_id = downstream_id
        self.agent_description = f"{agent_id}: upstream={upstream_id}, downstream={downstream_id}"

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
        if isinstance(action, list):
            action = action[0]
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

class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: ph.AgentID, retailer_id: ph.AgentID):
        super().__init__(agent_id)
        self.retailer_id = retailer_id  # 上游是 retailer

    def generate_messages(self, ctx: ph.Context):
        # 每步随机生成一个订单，发给 retailer
        order_size = np.random.randint(1, MAX_ORDER + 1)
        return [(self.retailer_id, OrderRequest1(order_size))]

    @ph.agents.msg_handler(Delivery1)
    def handle_delivery(self, ctx: ph.Context, message: ph.Message):
        # 顾客收到零售商发来的货物，不需要进一步处理
        return


# ====== 顶层环境 ======
class Network2Env(ph.PhantomEnv):
    def __init__(self):
        # 定义角色ID
        self.env_description = "A supply chain environment with 1 factory, 4 distributors, 4 wholesalers, and 4 retailers."
        factory_id = "FACTORY1"
        distributor_ids = [f"DISTRIBUTOR{i}" for i in range(1, 5)]
        wholesaler_ids = [f"WHOLESALER{i}" for i in range(1, 5)]
        retailer_ids = [f"RETAILER{i}" for i in range(1, 5)]
        customer_ids = [f"CUSTOMER{i}" for i in range(1, 5)]
        # customer_id = "CUSTOMER1"

        # 定义代理
        factory = SupplyChainAgent(factory_id, None, distributor_ids[0])  # 下游只用第一个分销商ID，实际可不影响

        distributors = [
            SupplyChainAgent(distributor_ids[i], factory_id, wholesaler_ids[i])
            for i in range(4)
        ]
        wholesalers = [
            SupplyChainAgent(wholesaler_ids[i], distributor_ids[i], retailer_ids[i])
            for i in range(4)
        ]
        retailers = [
            SupplyChainAgent(retailer_ids[i], wholesaler_ids[i], None)
            for i in range(4)
        ]

        # agents = [factory] + distributors + wholesalers + retailers
        # network = ph.Network(agents)
        customers = [
            CustomerAgent(customer_ids[i], retailer_id=retailer_ids[i])
            for i in range(4)
        ]
        # agents = customers + [factory] + distributors + wholesalers + retailers
        # customer = ph.Agent(customer_id)
        agents = customers + [factory] + distributors + wholesalers + retailers
        network = ph.Network(agents)
        # 建立连接
        # 前四条供应链
        for i in range(4):
            network.add_connection(factory_id, distributor_ids[i])
            network.add_connection(distributor_ids[i], wholesaler_ids[i])
            network.add_connection(wholesaler_ids[i], retailer_ids[i])
            network.add_connection(customer_ids[i], retailer_ids[i])
        # # 建立顾客到零售商的连接
        # for i in range(4):
        #     network.add_connection(customer_ids[i], retailer_ids[i])
        # 第五条供应链（交叉链）
        network.add_connection(factory_id, distributor_ids[3])
        network.add_connection(distributor_ids[3], wholesaler_ids[3])
        network.add_connection(wholesaler_ids[3], retailer_ids[2])
        # # 建立顾客到零售商的连接
        # for retailer_id in retailer_ids:
        #     network.add_connection(customer_id, retailer_id)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network)
    # def step(self, actions):
    #     # 每步为每个零售商生成顾客需求
    #     for retailer_id in [f"RETAILER{i}" for i in range(1, 5)]:
    #         customer_demand = np.random.randint(1, MAX_ORDER + 1)
    #         self.network.send(
    #             sender_id="CUSTOMER1",  # 这里不用加到 agent_name，只是标识
    #             receiver_id=retailer_id,
    #             payload=OrderRequest(customer_demand)
    #         )
    #     # 继续调用父类 step 处理 agent 行为
    #     state, reward, done, info, _ = super().step(actions)
        # # 补全 reward 字典，防止 KeyError
        # for agent_id in self.agents.keys():
        #     if agent_id not in reward:
        #         reward[agent_id] = 0
        # return state, reward, done, info, _