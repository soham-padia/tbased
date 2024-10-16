# src/reward.py

from src.parameters import MAX_TRANSACTION_THROUGHPUT, MAX_COMPUTATION_COST

def calculate_reward(consensus_success, malicious_node_ratio, transaction_throughput, computation_cost):
    """
    Calculate the reward based on multiple factors:
    - Consensus success: whether consensus was achieved in the blockchain.
    - Malicious node ratio: the proportion of malicious nodes in the network.
    - Transaction throughput: number of transactions processed per unit time.
    - Computation cost: the total computational cost incurred.

    The reward function aims to maximize consensus success and transaction throughput,
    minimize the malicious node ratio and computation cost.
    """

    # Weight factors for each component of the reward
    w_consensus = 0.5
    w_throughput = 0.3
    w_malicious = 0.1
    w_cost = 0.1

    # Normalize values to be between 0 and 1
    normalized_throughput = transaction_throughput / MAX_TRANSACTION_THROUGHPUT  # You need to define MAX_TRANSACTION_THROUGHPUT
    normalized_cost = computation_cost / MAX_COMPUTATION_COST  # You need to define MAX_COMPUTATION_COST

    # Ensure that higher malicious node ratio and computation cost reduce the reward
    reward = (
        w_consensus * consensus_success +
        w_throughput * normalized_throughput -
        w_malicious * malicious_node_ratio -
        w_cost * normalized_cost
    )

    return reward
