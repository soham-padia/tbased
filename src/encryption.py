# src/encryption.py

from Pyfhel import Pyfhel

class FHE:
    def __init__(self):
        self.HE = Pyfhel()
        # For BFV scheme with plaintext modulus bits t_bits=20
        self.HE.contextGen(scheme='BFV', n=2**14, t_bits=20)
        self.HE.keyGen()
       # Generate public and private keys

    def encrypt(self, plaintext):
        return self.HE.encryptInt(plaintext)  # Encrypt integer plaintext

    def decrypt(self, ciphertext):
        return self.HE.decryptInt(ciphertext)  # Decrypt to integer

    def add(self, ctxt1, ctxt2):
        return ctxt1 + ctxt2  # Homomorphic addition

    def multiply(self, ctxt1, ctxt2):
        return ctxt1 * ctxt2  # Homomorphic multiplication
