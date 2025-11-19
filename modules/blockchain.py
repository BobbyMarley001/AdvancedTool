from web3 import Web3
import json
import os

class BlockchainManager:
    def __init__(self):
        
        infura_url = "https://sepolia.infura.io/v3/YOUR-PROJECT-ID"  
        self.web3 = Web3(Web3.HTTPProvider(infura_url))


        self.private_key = "YOUR-PRIVATE-KEY"
        self.account = self.web3.eth.account.from_key(self.private_key)
        self.web3.eth.default_account = self.account.address


        contract_info_path = "contract_info.json"
        if not os.path.exists(contract_info_path):
            raise FileNotFoundError("فایل contract_info.json یافت نشد! لطفاً قرارداد را دیپلوی کنید.")
        with open(contract_info_path, "r") as f:
            contract_info = json.load(f)
        self.contract_address = contract_info["address"]
        self.abi = contract_info["abi"]


        self.contract = self.web3.eth.contract(address=self.contract_address, abi=self.abi)

    def log_to_blockchain(self, action, data):
        try:

            nonce = self.web3.eth.get_transaction_count(self.account.address)
            tx = self.contract.functions.storeLog(action, data).build_transaction({
                "from": self.account.address,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": self.web3.to_wei("20", "gwei"),
            })

            
            signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            return f"لاگ در بلاکچین ذخیره شد: {action} - {data}\nتراکنش: {tx_hash.hex()}"
        except Exception as e:
            return f"خطا در ذخیره در بلاکچین: {str(e)}"

    def get_logs_from_blockchain(self):
        try:
            log_count = self.contract.functions.getLogCount().call()
            logs = []
            for i in range(log_count):
                log = self.contract.functions.getLog(i).call()
                logs.append({"action": log[0], "data": log[1], "timestamp": log[2]})
            return logs
        except Exception as e:
            return f"خطا در دریافت لاگ‌ها از بلاکچین: {str(e)}"