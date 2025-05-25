from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

def build_sudoku_tokenizer(pad_vocab_to=0):
    if pad_vocab_to > 0:
        path = "gidd/tokenizers/sudoku_padded"
    else:
        path = "gidd/tokenizers/sudoku"
    
    tokenizer_json = "/tokenizer.json"
    
    digits = "123456789"
    core_vocab = list(digits)
    # special_tokens = ["0", "[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    special_tokens = ["[PAD]","0", "[UNK]", "[BOS]", "[EOS]"]
    tokens_to_pad_vocab_length = [f"[VPAD{i}]" for i in range(pad_vocab_to - len(core_vocab) - len(special_tokens))]
    vocab = core_vocab + special_tokens + tokens_to_pad_vocab_length

    tokenizer = Tokenizer(WordLevel({k: i for i, k in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="removed")
    tokenizer.save(path + tokenizer_json)

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path + tokenizer_json)
    fast_tokenizer.add_special_tokens({"pad_token": "[PAD]", "mask_token": "0", "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]"})
    fast_tokenizer.save_pretrained(path)

# build_sudoku_tokenizer()
# build_sudoku_tokenizer(pad_vocab_to=128)