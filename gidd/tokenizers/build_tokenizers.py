from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

def build_sudoku_tokenizer():
    # path = "gidd/tokenizers/sudoku"
    path = "gidd/tokenizers/sudoku_padded"
    tokenizer_json = "/tokenizer.json"
    
    digits = "123456789"
    # vocab = list(digits) + ["[PAD]", "0", "[UNK]", "[BOS]", "[EOS]"]
    core_vocab = list(digits)
    special_tokens = ["[PAD]", "0", "[UNK]", "[BOS]", "[EOS]"]
    padded_vocab_len = 128
    tokens_to_pad_vocab_length = [f"[VPAD{i}]" for i in range(padded_vocab_len - len(core_vocab) - len(special_tokens))]
    vocab = core_vocab + special_tokens + tokens_to_pad_vocab_length

    tokenizer = Tokenizer(WordLevel({k: i for i, k in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="removed")
    tokenizer.save(path + tokenizer_json)

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path + tokenizer_json)
    fast_tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "0", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"})
    fast_tokenizer.save_pretrained(path)

build_sudoku_tokenizer()