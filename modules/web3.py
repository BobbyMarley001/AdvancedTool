from web3 import Web3
import json

infura_url = "https://sepolia.infura.io/v3/YOUR-PROJECT-ID"  # جای YOUR-PROJECT-ID رو با Project ID خودت پر کن
web3 = Web3(Web3.HTTPProvider(infura_url))


private_key = "YOUR-PRIVATE-KEY"  
account = web3.eth.account.from_key(private_key)
web3.eth.default_account = account.address


bytecode = "YOUR-CONTRACT-BYTECODE"  
abi = json.loads('YOUR-CONTRACT-ABI')  

LogStorage = web3.eth.contract(abi=abi, bytecode=bytecode)

nonce = web3.eth.get_transaction_count(account.address)
tx = LogStorage.constructor().build_transaction({
    "from": account.address,
    "nonce": nonce,
    "gas": 2000000,
    "gasPrice": web3.to_wei("20", "gwei"),
})

signed_tx = web3.eth.account.sign_transaction(tx, private_key)
tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

contract_address = tx_receipt.contractAddress
print(f"قرارداد در آدرس {contract_address} دیپلوی شد")

with open("contract_info.json", "w") as f:
    json.dump({"abi": abi, "address": contract_address}, f)