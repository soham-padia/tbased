# src/trust.py

class TrustManager:
    def __init__(self, nodes):
        self.trust_scores = {node_id: 0.5 for node_id in nodes}  # Initialize with neutral trust

    def update_trust(self, node_id, outcome):
        if outcome == "valid":
            self.trust_scores[node_id] += 0.1
        else:
            self.trust_scores[node_id] -= 0.2
        # Clamp trust scores between 0 and 1
        self.trust_scores[node_id] = max(0, min(1, self.trust_scores[node_id]))

    def get_trust(self, node_id):
        return self.trust_scores.get(node_id, 0.5)

    def select_delegated_nodes(self, num_delegates):
        # Select nodes with the highest trust scores
        sorted_nodes = sorted(self.trust_scores.items(), key=lambda item: item[1], reverse=True)
        delegated_nodes = [node_id for node_id, trust in sorted_nodes[:num_delegates]]
        return delegated_nodes
