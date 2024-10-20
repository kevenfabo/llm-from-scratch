import os
import re
import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

def download_vocab(data_folder: str) -> None:
    # data source url 
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    
    # file path
    file_path = f"{data_folder}/the-verdict.txt"
    
    # download the file and fix certificates issues
    
    urllib.request.urlretrieve(url, file_path)
    
    return file_path

def read_vocab(data_folder: str) -> str:
    # corpus file path
    file_path = f"{data_folder}/the-verdict.txt"
    
    # read the downloaded vocab file
    with open(file_path, 'r') as f:
        text = f.read()
    
    # build the preprocessed raw data
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [
        item.strip() for item in preprocessed if item.strip()
    ]
    unique_words = set(preprocessed)
    vocab = {
        token:i for i, token in enumerate(unique_words)
    }
    
    return vocab

if __name__=="__main__":

    # create a data folder if it does not exist
    if not os.path.exists("data"):
        os.makedirs("data")
        
    # download the vocabulary file
    _ = download_vocab(data_folder="data")
    
    # read the vocabulary file
    text = read_vocab(data_folder="data")
    
    # print the first 1000 characters
    print(text[:1000])