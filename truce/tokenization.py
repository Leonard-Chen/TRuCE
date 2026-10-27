#!/usr/bin/env python3
import hashlib


class TinyTokenizer:
    '''
    仅用于离线流程测试，不能用于生成正式实验结果
    '''

    pad_token_id = 0
    unk_token_id = 100
    cls_token_id = 101
    sep_token_id = 102
    pad_token = '[PAD]'
    unk_token = '[UNK]'
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.special = {'[PAD]': 0, '[UNK]': 100, '[CLS]': 101,
                        '[SEP]': 102, '[unused0]': 1, '[unused1]': 2}

    def tokenize(self, text, **kwargs):
        return text.split() or [self.unk_token]

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            if tokens in self.special:
                return self.special[tokens]
            digest = hashlib.sha256(tokens.lower().encode('utf-8')).digest()
            return 103 + int.from_bytes(digest[:8], 'big') % (self.vocab_size - 103)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def encode(self, text, add_special_tokens=True, **kwargs):
        ids = self.convert_tokens_to_ids(self.tokenize(text))
        return self.build_inputs_with_special_tokens(ids) if add_special_tokens else ids

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.cls_token_id] + list(token_ids) + [self.sep_token_id]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def get_special_tokens_mask(self, token_ids, already_has_special_tokens=False):
        if already_has_special_tokens:
            return [int(token in (101, 102, 0)) for token in token_ids]
        return [1] + [0] * len(token_ids) + [1]

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return {'input_ids': ids, 'attention_mask': [1] * len(ids)}


def create_tokenizer(config):
    model = config.get('model', config)
    if model.get('encoder_backend', 'hf') == 'tiny':
        return TinyTokenizer(model.get('tiny_vocab_size', 4096))

    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model.get('encoder_name', 'bert-base-uncased'),
        revision=model.get('encoder_revision', 'main'),
        local_files_only=model.get('local_files_only', False),
        use_fast=True,
    )
