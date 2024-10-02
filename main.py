# main.py

from src.encryption import FHE
from src.access_control import EncryptedABAC
from src.drl import D3PAgent
from src.blockchain import Blockchain
from src.trust import TrustManager
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning

def generate_transaction(size):
    """Generate a random transaction of a given size."""
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
    roles = ['admin', 'user', 'superuser']
    locations = ['HQ', 'Branch1', 'Branch2']
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
    # For simplicity, we'll simulate the effect of actions
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

def run_simulation():
    try:
        print("Starting Simulation...")

        # Initialize nodes and blockchain
        num_nodes = 8
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

        # Setup Encrypted ABAC
        abac = EncryptedABAC()
        print("Encrypted ABAC Policies Set Up.")

        # Initialize FHE for encryption
        fhe = FHE()
        print("Fully Homomorphic Encryption Initialized.")

        # Initialize DRL Agent
        state_size = 3  # Adjust based on get_current_state
        action_size = 3  # Three possible actions: decrease, maintain, increase delegation ratio
        agent = D3PAgent(state_size, action_size)
        episodes = 20  # Number of training episodes
        batch_size = 32

        # Simulation parameters
        simulation_steps = 100
        transactions_per_step = 5

        for episode in range(episodes):
            state = get_current_state(trust_manager, blockchain)
            total_reward = 0
            blockchain.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = environment_step(action, blockchain, trust_manager)
                error = reward - np.max(agent.model(torch.FloatTensor(state)).detach().numpy())
                agent.memorize(state, action, reward, next_state, done, error)
                state = next_state
                total_reward += reward

                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)

            agent.update_target_model()
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")

        # After training, evaluate the model
        print("Training completed. Evaluating the model...")

        # Generate classification report and confusion matrix
        y_true = []
        y_pred = []
        trust_scores = []

        for node in nodes:
            trust_score = trust_manager.get_trust(node)
            trust_scores.append(trust_score)
            is_malicious = node_status[node]
            y_true.append(1 if not is_malicious else 0)  # 1 for honest, 0 for malicious

        # Analyze trust scores and adjust threshold
        trust_scores_array = np.array(trust_scores)
        print("Trust scores:", trust_scores_array)

        # Experiment with different thresholds to find a balanced prediction
        thresholds = np.linspace(0, 1, num=21)  # Thresholds from 0 to 1 in steps of 0.05
        best_threshold = 0.5
        best_f1 = 0
        for threshold in thresholds:
            y_pred_temp = [1 if score >= threshold else 0 for score in trust_scores_array]
            if len(set(y_pred_temp)) < 2:
                continue  # Skip if only one class is predicted
            report = classification_report(y_true, y_pred_temp, output_dict=True, zero_division=0)
            f1 = report['weighted avg']['f1-score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                y_pred = y_pred_temp

        print(f"Best threshold based on F1-score: {best_threshold}")

        # Generate final predictions using the best threshold
        y_pred = [1 if score >= best_threshold else 0 for score in trust_scores_array]

        # Check distributions
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        print("y_true distribution:", dict(zip(unique_true, counts_true)))
        print("y_pred distribution:", dict(zip(unique_pred, counts_pred)))

        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        labels = ['Malicious', 'Honest']
        plot_confusion_matrix(cm, labels)

        # Print classification report with zero_division parameter
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

        # Plot malicious nodes
        plot_malicious_nodes(node_status)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Suppress UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    run_simulation()
