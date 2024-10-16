# main.py

from src.encryption import FHE
from src.access_control import EncryptedABAC
from src.drl import D3PAgent
from src.blockchain import Blockchain
from src.trust import TrustManager
from src.parameters import *  # Import all parameters

import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import sys
import traceback

ATTRIBUTE_VALUE_MAPPING = {
    'role': {'admin': 1, 'user': 2, 'superuser': 3},
    'location': {'HQ': 1, 'Branch1': 2, 'Branch2': 3}
}

# Reverse mapping for printing purposes (optional)
REVERSE_ATTRIBUTE_VALUE_MAPPING = {
    'role': {1: 'admin', 2: 'user', 3: 'superuser'},
    'location': {1: 'HQ', 2: 'Branch1', 3: 'Branch2'}
}

def generate_transaction():
    """Generate a random transaction of average size."""
    size = TRANSACTION_SIZE_AVG  # Use average transaction size from parameters
    return random.getrandbits(size * 8)  # Simulate transaction as a bitstring

def plot_confusion_matrix(cm, labels):
    """Plot the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def plot_malicious_nodes(node_status):
    """Plot the graph for malicious node detection."""
    G = nx.Graph()

    for node, is_malicious in node_status.items():
        G.add_node(node, malicious=is_malicious)

    # Add edges between nodes (assuming a fully connected graph for simplicity)
    for i in range(len(node_status)):
        for j in range(i + 1, len(node_status)):
            G.add_edge(list(node_status.keys())[i], list(node_status.keys())[j])

    colors = ['red' if is_malicious else 'green' for is_malicious in node_status.values()]
    nx.draw(G, with_labels=True, node_color=colors, font_weight='bold', font_size=10)
    plt.title('Malicious Node Detection')
    plt.show()

def get_node_attributes(node_id):
    """Retrieve attributes for a given node."""
    # For simplicity, assign random attributes
    roles = ['admin','admin', 'user', 'superuser']
    locations = ['HQ','HQ', 'Branch1', 'Branch2']
    return {
        'role': random.choice(roles),
        'location': random.choice(locations)
    }

def get_current_state(trust_manager, blockchain):
    """Define the state representation for the DRL agent."""
    # Example state: average trust value, number of pending transactions, etc.
    avg_trust = np.mean(list(trust_manager.trust_scores.values()))
    pending_tx = len(blockchain.pending_transactions)
    num_blocks = len(blockchain.chain)
    state = np.array([avg_trust, pending_tx, num_blocks], dtype=np.float32)
    return state

def environment_step(action, blockchain, trust_manager):
    """Apply action to the environment and return the next state, reward, and done flag."""
    # Actions could be adjusting block size, block interval, delegation ratio
    # action 0: decrease delegation ratio
    # action 1: maintain delegation ratio
    # action 2: increase delegation ratio

    if action == 0:
        blockchain.delegation_ratio = max(0.1, blockchain.delegation_ratio - 0.1)
    elif action == 2:
        blockchain.delegation_ratio = min(0.9, blockchain.delegation_ratio + 0.1)

    # Perform mining with the updated delegation ratio
    blockchain.mine()

    # Calculate reward based on consensus success and security metrics
    consensus_success = blockchain.consensus_success
    malicious_node_ratio = blockchain.get_malicious_node_ratio()
    reward = consensus_success - malicious_node_ratio  # Reward higher consensus success and lower malicious nodes

    # Get next state
    next_state = get_current_state(trust_manager, blockchain)

    # Determine if the simulation should end (for example, after a certain number of steps)
    done = blockchain.current_step >= blockchain.max_steps

    return next_state, reward, done

def run_simulation(config):
    try:
        print("Starting Simulation...")

        # Initialize nodes and blockchain
        num_nodes = TOTAL_NODES  # Use TOTAL_NODES from parameters
        nodes = [f"Node{i+1}" for i in range(num_nodes)]
        node_status = {}  # Keeps track of whether a node is malicious
        for node in nodes:
            # Randomly assign some nodes as malicious (20% chance)
            node_status[node] = random.random() < 0.2  # 20% chance of being malicious

        # Initialize Trust Manager
        trust_manager = TrustManager(nodes)

        # Initialize Blockchain with the trust_manager
        blockchain = Blockchain(nodes, trust_manager)
        print("Blockchain Initialized with Trust Management.")

        # Initialize FHE for encryption
        fhe = FHE()
        print("Fully Homomorphic Encryption Initialized.")

        # Setup Encrypted ABAC
        abac = EncryptedABAC()
        abac.fhe = fhe  # Ensure ABAC uses the same FHE instance
        print("Encrypted ABAC Policies Set Up.")

        # Define a policy in the ABAC system
        policy_id = "Policy1"
        policy_attributes = {'role': 'admin', 'location': 'HQ'}

        # Map policy attributes to integers
        mapped_policy_attributes = {k: ATTRIBUTE_VALUE_MAPPING[k][v] for k, v in policy_attributes.items()}
        abac.add_policy(policy_id, mapped_policy_attributes)
        print(f"Policy {policy_id} added with attributes: {policy_attributes}")

        # Generate attributes for each node and add them to ABAC
        access_results = {}  # To keep track of access results
        for node in nodes:
            attributes = get_node_attributes(node)
            # Map attributes to integers
            mapped_attributes = {k: ATTRIBUTE_VALUE_MAPPING[k][v] for k, v in attributes.items()}
            abac.add_user_attributes(node, mapped_attributes)
            access_granted = abac.evaluate_policy(node, policy_id)
            access_results[node] = access_granted
            status = "GRANTED" if access_granted else "DENIED"
            print(f"Access {status} for node {node} with attributes: {attributes}")

        # After processing all nodes, print the final results
        total_granted = sum(1 for granted in access_results.values() if granted)
        total_denied = len(access_results) - total_granted
        print(f"\nTotal nodes granted access: {total_granted}")
        print(f"Total nodes denied access: {total_denied}")

        # Initialize DRL Agent
        state_size = 3  # Adjust based on get_current_state
        action_size = 3  # Three possible actions: decrease, maintain, increase delegation ratio
        agent = D3PAgent(state_size, action_size)
        episodes = 100  # Number of training episodes
        batch_size = BATCH_SIZE  # Use BATCH_SIZE from parameters

        # Simulation parameters
        simulation_steps = 100
        transactions_per_step = 5

        for episode in range(episodes):
            state = get_current_state(trust_manager, blockchain)
            total_reward = 0
            blockchain.reset()
            done = False

            while not done:
                # Generate transactions
                for _ in range(transactions_per_step):
                    transaction = generate_transaction()
                    blockchain.add_transaction(transaction)

                action = agent.act(state)
                next_state, reward, done = environment_step(action, blockchain, trust_manager)

                # Memorize the experience
                agent.memorize(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if len(agent.memory.buffer) > batch_size:
                    agent.replay(batch_size)

            agent.update_target_model()
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

        # After training, evaluate the model
        print("Training completed. Evaluating the model...")

        # Analyze trust scores and adjust threshold
        trust_scores = [trust_manager.get_trust(node) for node in nodes]
        print("Trust scores:", trust_scores)

        # Try different thresholds to find the best one
        thresholds = [TRUST_THRESHOLD - 0.2, TRUST_THRESHOLD - 0.1, TRUST_THRESHOLD, TRUST_THRESHOLD + 0.1]
        best_threshold = TRUST_THRESHOLD
        best_f1 = 0

        for threshold in thresholds:
            y_true = []
            y_pred = []

            for node in nodes:
                trust_score = trust_manager.get_trust(node)
                is_malicious = node_status[node]
                y_true.append(1 if not is_malicious else 0)  # 1 for honest, 0 for malicious
                y_pred.append(1 if trust_score >= threshold else 0)

            unique_pred = np.unique(y_pred)
            if len(unique_pred) < 2:
                print(f"Warning: Only one class predicted with threshold {threshold}")
                continue  # Skip this threshold

            # Generate classification report
            report = classification_report(y_true, y_pred, target_names=['Malicious', 'Honest'], zero_division=0, output_dict=True)
            f1_score = report['weighted avg']['f1-score']
            print(f"Threshold: {threshold}, F1-Score: {f1_score:.4f}")

            if f1_score > best_f1:
                best_f1 = f1_score
                best_threshold = threshold
                best_y_pred = y_pred
                best_cm = confusion_matrix(y_true, y_pred)

        print(f"\nBest Threshold: {best_threshold}, Best F1-Score: {best_f1:.4f}")

        # Plot confusion matrix with the best threshold
        labels = ['Malicious', 'Honest']
        plot_confusion_matrix(best_cm, labels)

        # Print classification report for the best threshold
        print("Classification Report:")
        print(classification_report(y_true, best_y_pred, target_names=labels, zero_division=0))

        # Plot malicious nodes
        plot_malicious_nodes(node_status)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config = sys.argv[1]
    else:
        config = 'ABAC-TDCB-D3P-NMA'  # Default configuration
    run_simulation(config)
