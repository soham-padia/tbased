# src/parameters.py

# Parameters
TRANSACTION_SIZE_AVG = 256  # Average size of the transaction in bytes
BLOCK_HEADER_SIZE = 64       # Size of the block header in bytes
TOTAL_NODES = 10             # Total number of nodes in the blockchain
SIGNATURE_VERIFICATION_COST = 0.01  # Computational expense for verifying signatures
BLOCK_INTERVAL = 10          # Consecutive block interval for blockchain finality in seconds
DATA_TRANSMISSION_SPEED = 1000  # Data transmission speed in KB/s
MAX_BLOCK_INTERVAL = 60      # Maximum interval between blocks in seconds
BATCH_SIZE = 32              # Size of the batch
BLOCK_SIZE_MB = 4            # Size of the Block in MB
MAC_CREATION_VERIFICATION_COST = 0.005  # Computational expense for creating and verifying the MAC
NODE_COMPUTING_CAPACITY = 10  # Computing capacity of node i (arbitrary unit)
TRUST_THRESHOLD = 0.5        # Trust threshold for identifying malicious nodes
