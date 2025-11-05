import torch
from torch.utils.data import TensorDataset, random_split
import xml.etree.ElementTree as ET

class CharTokenizer:
    def __init__(self, texts, add_special_tokens=True):
        chars = set()
        for text in texts:
            chars.update(list(text))
        chars = sorted(list(chars))

        self.special_tokens = ['<pad>', '<unk>'] if add_special_tokens else []
        self.chars = self.special_tokens + chars
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi.get(c, self.stoi.get('<unk>', 1)) for c in text]

    def decode(self, indices):
        return ''.join([self.itos.get(i, '<unk>') for i in indices])

import xml.etree.ElementTree as ET

def parse_iwslt_xml(src_xml, tgt_xml):
    def get_segments(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        segments = []
        # 遍历所有 doc
        for doc in root.findall(".//doc"):
            for seg in doc.findall("seg"):
                if seg.text and seg.text.strip():
                    segments.append(seg.text.strip())
        return segments

    src_texts = get_segments(src_xml)
    tgt_texts = get_segments(tgt_xml)

    print(f"DEBUG: Loaded {len(src_texts)} source sentences, {len(tgt_texts)} target sentences")

    if len(src_texts) == 0 or len(tgt_texts) == 0:
        raise ValueError("No data found. Check your XML paths and parsing logic.")

    if len(src_texts) != len(tgt_texts):
        # 保证句子数一致
        min_len = min(len(src_texts), len(tgt_texts))
        src_texts = src_texts[:min_len]
        tgt_texts = tgt_texts[:min_len]

    return src_texts, tgt_texts



def build_dataset_from_iwslt(src_xml, tgt_xml, seq_len=128, val_ratio=0.1, test_ratio=0.1):

    src_texts, tgt_texts = parse_iwslt_xml(src_xml, tgt_xml)
    print(f"Loaded {len(src_texts)} source sentences, {len(tgt_texts)} target sentences")
    if len(src_texts) == 0 or len(tgt_texts) == 0:
        raise ValueError("No data found. Check your XML paths and parsing logic.")

    tokenizer = CharTokenizer(src_texts + tgt_texts)

    def pad_seq(seq):
        if len(seq) > seq_len:
            return seq[:seq_len]
        return seq + [tokenizer.stoi['<pad>']] * (seq_len - len(seq))

    src_encoded = torch.tensor([pad_seq(tokenizer.encode(s)) for s in src_texts], dtype=torch.long)
    tgt_encoded = torch.tensor([pad_seq(tokenizer.encode(t)) for t in tgt_texts], dtype=torch.long)

    dataset = TensorDataset(src_encoded, tgt_encoded)

    # 划分训练/验证/测试集
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    return train_set, val_set, test_set, tokenizer
