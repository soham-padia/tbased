# src/blockchain.py

import time
from .block import Block
from .parameters import NODE_COMPUTING_CAPACITY

class Blockchain:
    def __init__(self, nodes, trust_manager):
        self.chain = []
        self.pending_transactions = []
        self.nodes = nodes
        self.trust_manager = trust_manager
        self.current_step = 0
        self.max_steps = 100
        self.delegation_ratio = 0.5  # Initial delegation ratio
        self.consensus_success = 0  # Initialize consensus success metric
        self.create_genesis_block()

    def get_transaction_throughput(self):
        """
        Calculate the transaction throughput.
        For simplicity, return the number of transactions in the last block.
        """
        if len(self.chain) > 1:
            last_block = self.chain[-1]
            return len(last_block.transactions)
        else:
            return 0

    def get_computation_cost(self):
        """
        Calculate the computational cost.
        For simplicity, assume each node incurs a fixed computational cost per mining operation.
        """
        computation_cost_per_node = NODE_COMPUTING_CAPACITY  # From parameters.py
        num_delegated_nodes = max(1, int(len(self.nodes) * self.delegation_ratio))
        total_cost = computation_cost_per_node * num_delegated_nodes
        return total_cost

    def create_genesis_block(self):
        genesis_block = Block(0, [], time.time(), "0")
        self.chain.append(genesis_block)

    def add_transaction(self, transaction):
        self.pending_transactions.append(transaction)

    def mine(self):
        self.current_step += 1  # Increment the step counter

        # Determine the number of delegates based on the delegation ratio
        num_delegates = max(1, int(len(self.nodes) * self.delegation_ratio))
        delegated_nodes = self.trust_manager.select_delegated_nodes(num_delegates)

        # Simulate consensus among delegated nodes
        consensus_reached = self.simulate_consensus(delegated_nodes)

        if consensus_reached:
            last_block = self.chain[-1]
            new_block = Block(index=last_block.index + 1,
                              transactions=self.pending_transactions,
                              timestamp=time.time(),
                              previous_hash=last_block.hash)
            self.chain.append(new_block)
            self.pending_transactions = []
            self.consensus_success = 1  # Successful consensus
            # Update trust scores positively
            for node_id in delegated_nodes:
                self.trust_manager.update_trust(node_id, "valid")
        else:
            self.consensus_success = 0  # Consensus failed
            # Update trust scores negatively
            for node_id in delegated_nodes:
                self.trust_manager.update_trust(node_id, "invalid")

    def simulate_consensus(self, delegated_nodes):
        # For simplicity, assume consensus is reached if more than 2/3 nodes are honest
        honest_nodes = [node_id for node_id in delegated_nodes if self.trust_manager.get_trust(node_id) > 0.5]
        return len(honest_nodes) >= (2 * len(delegated_nodes) // 3)

    def reset(self):
        self.chain = self.chain[:1]  # Keep only the genesis block
        self.pending_transactions = []
        self.current_step = 0
        self.consensus_success = 0

    def get_malicious_node_ratio(self):
        malicious_nodes = [node_id for node_id in self.nodes if self.trust_manager.get_trust(node_id) < 0.5]
        return len(malicious_nodes) / len(self.nodes)
