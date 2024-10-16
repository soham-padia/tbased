# src/encryption.py
import numpy as np
from Pyfhel import Pyfhel

class FHE:
    def __init__(self):
        self.HE = Pyfhel()
        # For BFV scheme with plaintext modulus t=65537
        self.HE.contextGen(scheme='BFV', n=2**14, t=65537)
        self.HE.keyGen()
        # Generate public and private keys

    def encrypt(self, plaintext):
        # Convert plaintext integer to NumPy array
        plaintext_array = np.array([plaintext], dtype=np.int64)
        return self.HE.encryptInt(plaintext_array)


    def decrypt(self, ciphertext):
        decrypted_array = self.HE.decryptInt(ciphertext)
        return int(decrypted_array[0])


    def add(self, ctxt1, ctxt2):
        return ctxt1 + ctxt2  # Homomorphic addition

    def multiply(self, ctxt1, ctxt2):
        return ctxt1 * ctxt2  # Homomorphic multiplication

