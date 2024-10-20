import re 
from vocab import read_vocab

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {
            i:s for s, i in vocab.items()
        }
        
    def encode(self, text):
        # process input text into token ids 
        
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        
        return [self.str_to_int[token] for token in preprocessed]
        
    def decode(self, ids):
        text = " ".join(
            [self.int_to_str[i] for i in ids]
        )
        cleaned_text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return cleaned_text
        
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {
            i:s for s, i in vocab.items()
        }
        
        # add special tokens: "<|endoftext|>", "<|unk|>"
        special_tokens = ["<|endoftext|>", "<|unk|>"] # can be an input parameter
        for i, token in enumerate(special_tokens):
            max_id = max(self.str_to_int.values())
            if token not in self.str_to_int:
                self.str_to_int[token] = max_id + i
                self.int_to_str[max_id + i] = token

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        return [self.str_to_int[token] for token in preprocessed]
        
    def decode(self, ids):
        text = " ".join(
            [self.int_to_str[i] for i in ids]
        )
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text
        
            
if __name__=="__main__":
    
    # get the corpus vocabulary
    vocab  = read_vocab(data_folder="data")
    
    tokenizer_v1 = SimpleTokenizerV1(
        vocab=vocab
    )
    
    text_1 = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
        
    print(f"Original text: {text_1}")

    encoded_text_v1 = tokenizer_v1.encode(text_1)
    print(f"Encoded text: {encoded_text_v1}")
    
    decoded_encoded_text_v1 = tokenizer_v1.decode(encoded_text_v1)
    print(f"Decoded text from encoded text: {decoded_encoded_text_v1}")
    
    error_text = "Hello, World!"
    try:
        encoded_error_text = tokenizer_v1.encode(error_text)
    except Exception as e:
        print(f"Error: {e}")    
     
    # leveraging the new tokenizer   
    tokenizer_v2 = SimpleTokenizerV2(
        vocab=vocab
    )
    encoded_text_v2 = tokenizer_v2.encode(error_text)
    decoded_encoded_error_text_v2 = tokenizer_v2.decode(encoded_text_v2)
    print(f"Decoded text from encoded error text: {decoded_encoded_error_text_v2}")
    