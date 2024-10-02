# src/access_control.py

from .encryption import FHE

class EncryptedABAC:
    def __init__(self):
        self.fhe = FHE()
        self.encrypted_policies = {}
        self.encrypted_attributes = {}

    def add_policy(self, policy_id, attributes):
        encrypted_attrs = {k: self.fhe.encrypt(v) for k, v in attributes.items()}
        self.encrypted_policies[policy_id] = encrypted_attrs

    def add_user_attributes(self, user_id, attributes):
        encrypted_attrs = {k: self.fhe.encrypt(v) for k, v in attributes.items()}
        self.encrypted_attributes[user_id] = encrypted_attrs

    def evaluate_policy(self, user_id, policy_id):
        user_attrs = self.encrypted_attributes[user_id]
        policy_attrs = self.encrypted_policies[policy_id]
        result = True
        for attr, enc_policy_value in policy_attrs.items():
            if attr in user_attrs:
                enc_user_value = user_attrs[attr]
                # Perform encrypted comparison (not directly possible)
                # For simplicity, assume binary attributes and use homomorphic operations
                diff = self.fhe.add(enc_policy_value, self.fhe.multiply(enc_user_value, -1))
                decrypted_diff = self.fhe.decrypt(diff)
                if decrypted_diff != 0:
                    result = False
                    break
            else:
                result = False
                break
        return result
