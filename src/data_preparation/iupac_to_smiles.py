import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading


request_count = 0
request_count_lock = threading.Lock()

proxies = [
    "194.31.162.106:7622:pqphyzgx:vumw8d6sr2th",
    "206.41.175.187:6400:pqphyzgx:vumw8d6sr2th",
    "194.116.250.178:6636:pqphyzgx:vumw8d6sr2th",
    "171.22.251.123:5653:pqphyzgx:vumw8d6sr2th",
    "156.238.10.167:5249:pqphyzgx:vumw8d6sr2th",
    "185.242.95.15:6356:pqphyzgx:vumw8d6sr2th",
    "206.41.172.26:6586:pqphyzgx:vumw8d6sr2th",
    "171.22.248.83:5975:pqphyzgx:vumw8d6sr2th",
    "206.41.168.23:6688:pqphyzgx:vumw8d6sr2th",
    "154.95.0.46:6299:pqphyzgx:vumw8d6sr2th"
]
def format_proxy(index):
    proxy_info = proxies[index].split(":")
    return f"http://{proxy_info[2]}:{proxy_info[3]}@{proxy_info[0]}:{proxy_info[1]}"

def iupac_to_smiles(iupac_name, proxy):
    url = f'https://opsin.ch.cam.ac.uk/opsin/{iupac_name}.json'
    proxy_dict = {"http": proxy, "https": proxy}
    response = requests.get(url, proxies=proxy_dict)

    with request_count_lock:
        global request_count
        request_count += 1
        print(f"API Request Number: {request_count}")

    if response.status_code == 200:
        return response.json().get('smiles', "")
    else:
        return ""

def process_row(row, proxy):
    iupac_name = row['Components']
    return iupac_to_smiles(iupac_name, proxy)

def process_csv(input_csv, output_csv, num_threads=10):
    df = pd.read_csv(input_csv, delimiter=',')
    df = df.head(8000)  #On limite à 21K3

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        proxy_indices = [i % len(proxies) for i in range(num_threads)]
        tasks = [executor.submit(process_row, row, format_proxy(proxy_indices[i % num_threads]))
                 for i, row in enumerate(df.to_dict('records'))]
        smiles_list = [task.result() for task in tasks]

    df['SMILES'] = smiles_list
    df.to_csv(output_csv, index=False)
    print("Fini", output_csv)

print(iupac_to_smiles('Ethylammonium nitrate',0))

process_csv('/Users/ethanabimelech/PycharmProjects/Ionic-Liquid-Prediction/data/eleccond2.csv', 'electricalsmiled2.csv')
