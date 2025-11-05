import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model import TransformerEncoder, TransformerDecoder, generate_causal_mask, create_padding_mask
from data import build_dataset_from_iwslt
from utils import set_seed, save_checkpoint, plot_curves

def train_seq2seq(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    train_set, val_set, test_set, tokenizer = build_dataset_from_iwslt(
        config['src_xml'], config['tgt_xml'], seq_len=config['seq_len']
    )
    print(f"Loaded {len(train_set)} training samples, {len(val_set)} validation samples")
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    vocab_size = len(tokenizer.chars)
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_len=config['seq_len'],
        dropout=config['dropout']
    ).to(device)

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_layers=config['n_layers'],
        num_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_len=config['seq_len'],
        dropout=config['dropout']
    ).to(device)

    # 加载训练好的 checkpoint
    checkpoint = torch.load('results/seq2seq_epoch10.pt', map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=config['lr'], weight_decay=0.01)

    def lr_lambda(step):
        warmup_steps = config.get('warmup_steps', 200)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, 0.5 ** ((step - warmup_steps) / 1000.0))

    scheduler = LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.stoi['<pad>'])

    train_losses, val_losses = [], []
    global_step = 0

    for epoch in range(config['epochs']):
        encoder.train()
        decoder.train()
        epoch_train_loss = 0.0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = create_padding_mask(src)  # (B,1,1,T_src)

            memory, _ = encoder(src, mask=src_mask)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            tgt_mask = generate_causal_mask(tgt_input.size(1), device=device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(src.size(0), -1, -1)  # (B, T_tgt, T_tgt)
            memory_mask = create_padding_mask(src)  # (B,1,1,T_src)

            logits, _ = decoder(tgt_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_labels.contiguous().view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.get('grad_clip', 1.0))
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            global_step += 1

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        encoder.eval()
        decoder.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                src_mask = create_padding_mask(src)
                memory, _ = encoder(src, mask=src_mask)

                tgt_input = tgt[:, :-1]
                tgt_mask = generate_causal_mask(tgt_input.size(1), device=device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(src.size(0), -1, -1)
                memory_mask = create_padding_mask(src)

                logits, _ = decoder(tgt_input, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt[:, 1:].contiguous().view(-1))
                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['epochs']} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

        # 保存 checkpoint
        os.makedirs(config['save_dir'], exist_ok=True)
        save_checkpoint({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, os.path.join(config['save_dir'], f"seq2seq_epoch{epoch+1}.pt"))

    # 绘制训练曲线
    plot_curves(train_losses, val_losses, os.path.join(config['save_dir'], 'train_curves_seq2seq.png'))

    # 生成示例
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src, _ = train_set[0]  # 从验证集取样
        src = src.unsqueeze(0).to(device)
        memory, _ = encoder(src, mask=create_padding_mask(src))
        start_token_id = tokenizer.stoi.get('<sos>', 0)  # 或者你训练时使用的开始符
        ys = decoder.greedy_generate(memory, start_token_id=start_token_id,
                                     max_len=50,
                                     eos_token_id=tokenizer.stoi.get('<eos>', None),
                                     device=device)
        print('generated:', tokenizer.decode(ys[0].cpu()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = {
        'src_xml': '../IWSLT17.TED.tst2010.en-de.en.xml',
        'tgt_xml': '../IWSLT17.TED.tst2010.en-de.de.xml',
        'seed': args.seed,
        'seq_len': 128,
        'batch_size': 32,
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'lr': 3e-4,
        'epochs': 100,
        'grad_clip': 1.0,
        'warmup_steps': 200,
        'save_dir': 'results',
        'max_gen_len': 50
    }

    train_seq2seq(config)
