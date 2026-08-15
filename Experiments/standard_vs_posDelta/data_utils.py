import os
import re
import csv
import math
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


OUT = Path("/kaggle/working/standard_vs_posdelta")
OUT.mkdir(parents=True, exist_ok=True)


class Config:
    vocab_size = 500
    d_model = 128
    n_heads = 2
    n_hops = 3
    d_ff = 256
    max_len = 200
    dropout = 0.1
    label_smooth = 0.1

    # PosDelta
    pos_delta_init = 0.02
    pos_delta_rank = 4

    num_classes = None
    batch_size = 64
    epochs = 60
    patience = 20
    lr = 5e-4
    weight_decay = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"


SEEDS = [42, 43, 44, 45, 46]

BABI_DIR = (
    "/kaggle/input/datasets/roblexnana/"
    "the-babi-tasks-for-nlp-qa-system/tasks_1-20_v1-2/en-10k"
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())


class TokenizerCLS:
    """Tokenizer with <cls> prepended."""
    def __init__(self):
        self.w2i = {"<pad>": 0, "<unk>": 1, "<cls>": 2}
        self.vocab_size = 3

    def build(self, texts, maxv=None):
        ctr = Counter()
        for text in texts:
            ctr.update(tokenize(text))

        words = ctr.most_common(maxv - 3) if maxv else ctr.most_common()
        for word, _ in words:
            if word not in self.w2i:
                self.w2i[word] = len(self.w2i)

        self.vocab_size = len(self.w2i)

    def encode(self, text, maxlen):
        toks = tokenize(text)[: maxlen - 1]
        ids = [2] + [self.w2i.get(word, 1) for word in toks]
        mask = [1] * len(ids)

        pad = maxlen - len(ids)
        if pad > 0:
            ids += [0] * pad
            mask += [0] * pad

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )


class TokenizerPlain:
    """Tokenizer without <cls>, used by PosDelta."""
    def __init__(self):
        self.w2i = {"<pad>": 0, "<unk>": 1}
        self.vocab_size = 2

    def build(self, texts, maxv=None):
        ctr = Counter()
        for text in texts:
            ctr.update(tokenize(text))

        words = ctr.most_common(maxv - 2) if maxv else ctr.most_common()
        for word, _ in words:
            if word not in self.w2i:
                self.w2i[word] = len(self.w2i)

        self.vocab_size = len(self.w2i)

    def encode(self, text, maxlen):
        toks = tokenize(text)[:maxlen]
        ids = [self.w2i.get(word, 1) for word in toks]
        mask = [1] * len(ids)

        pad = maxlen - len(ids)
        if pad > 0:
            ids += [0] * pad
            mask += [0] * pad

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
        )


class TextDataset(Dataset):
    def __init__(self, samples, label_to_id, tokenizer, maxlen):
        self.samples = samples
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        ids, mask = self.tokenizer.encode(sample["text"], self.maxlen)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "label": torch.tensor(
                self.label_to_id[sample["label"]],
                dtype=torch.long,
            ),
        }


def pick_vocab_size(n_train_samples, task_name):
    if "bAbI" in task_name or "qa" in task_name.lower():
        return 200
    if n_train_samples < 5000:
        return 5000
    return 30000


def init_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_mean(x, mask):
    mask_f = mask.unsqueeze(-1).float()
    return (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)


def sinusoidal_pe(length, dim, base=10000.0):
    pe = torch.zeros(length, dim)
    positions = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float)
        * (-math.log(base) / dim)
    )

    pe[:, 0::2] = torch.sin(positions * div)

    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(positions * div)
    else:
        pe[:, 1::2] = torch.cos(positions * div[: dim // 2])

    return pe


def build_pos_delta(
    length,
    d_in,
    d_out,
    rank,
    base_row=10000.0,
    base_col=7919.0,
):
    """
    Build a fixed, block-partitioned rank-k positional ΔW tensor.
    Shape: [length, d_in, d_out].
    """
    if d_in % rank != 0 or d_out % rank != 0:
        raise ValueError(
            f"rank={rank} must divide both dimensions "
            f"(d_in={d_in}, d_out={d_out})"
        )

    block_in = d_in // rank
    block_out = d_out // rank

    pe_row = sinusoidal_pe(length, d_in, base=base_row)
    pe_col = sinusoidal_pe(length, d_out, base=base_col)

    pe_row_blocks = pe_row.view(length, rank, block_in)
    pe_col_blocks = pe_col.view(length, rank, block_out)

    delta = torch.zeros(length, d_in, d_out)

    for i in range(rank):
        block = torch.einsum(
            "lp,lq->lpq",
            pe_row_blocks[:, i, :],
            pe_col_blocks[:, i, :],
        )

        delta[
            :,
            i * block_in : (i + 1) * block_in,
            i * block_out : (i + 1) * block_out,
        ] = block

    return delta * (1.0 / math.sqrt(d_in))


def parse_babi(path):
    samples = []

    with open(path) as f:
        context = {}

        for line in f:
            line = line.strip()
            if not line:
                continue

            line_id, text = line.split(" ", 1)
            line_id = int(line_id)

            if line_id == 1:
                context = {}

            if "\t" in text:
                question, answer, _ = text.split("\t")

                samples.append(
                    {
                        "text": " ".join(
                            context[k] for k in sorted(context)
                        )
                        + " "
                        + question.strip(),
                        "label": answer.strip(),
                    }
                )
            else:
                context[line_id] = text.strip()

    return samples


def load_babi(babi_dir):
    tasks = [
        ("qa1_single-supporting-fact", "bAbI-qa1"),
        ("qa2_two-supporting-facts", "bAbI-qa2"),
        ("qa3_three-supporting-facts", "bAbI-qa3"),
        ("qa15_basic-deduction", "bAbI-qa15"),
        ("qa16_basic-induction", "bAbI-qa16"),
        ("qa19_path-finding", "bAbI-qa19"),
    ]

    datasets = {}

    for prefix, name in tasks:
        train_file = os.path.join(babi_dir, f"{prefix}_train.txt")
        test_file = os.path.join(babi_dir, f"{prefix}_test.txt")

        if os.path.exists(train_file) and os.path.exists(test_file):
            datasets[name] = {
                "train": parse_babi(train_file),
                "test": parse_babi(test_file),
            }

    return datasets


def build_loaders(
    variant,
    train_samples,
    test_samples,
    label_to_id,
    vocab_cap,
    cfg,
    seed,
):
    tokenizer_cls = (
        TokenizerCLS if variant == "Standard" else TokenizerPlain
    )

    tokenizer = tokenizer_cls()
    tokenizer.build(
        [s["text"] for s in train_samples + test_samples],
        maxv=vocab_cap,
    )

    cfg.vocab_size = tokenizer.vocab_size

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        TextDataset(
            train_samples,
            label_to_id,
            tokenizer,
            cfg.max_len,
        ),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        TextDataset(
            test_samples,
            label_to_id,
            tokenizer,
            cfg.max_len,
        ),
        shuffle=False,
        **loader_kwargs,
    )

    return tokenizer, train_loader, test_loader


def train_model(model, train_loader, test_loader, cfg, name):
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * cfg.epochs * len(train_loader)),
        cfg.epochs * len(train_loader),
    )

    best_acc = 0.0
    best_state = None
    no_improvement = 0

    history = {
        "tr_loss": [],
        "te_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for batch in train_loader:
            ids = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)
            labels = batch["label"].to(cfg.device)

            optimizer.zero_grad()

            loss = F.cross_entropy(
                model(ids, mask),
                labels,
                label_smoothing=cfg.label_smooth,
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batches += 1

        model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    batch["input_ids"].to(cfg.device),
                    batch["attention_mask"].to(cfg.device),
                )

                predictions.extend(
                    logits.argmax(1).cpu().numpy()
                )
                labels.extend(batch["label"].numpy())

        test_acc = float(
            np.mean(
                np.asarray(predictions) == np.asarray(labels)
            )
        )

        average_loss = total_loss / batches

        history["tr_loss"].append(average_loss)
        history["te_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, history


COLORS = {
    "Standard": "#888780",
    "PosDelta": "#2b6cb0",
}


def plot_training_curves(histories, task_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    fig.suptitle(
        f"Standard vs PosDelta — Training Curves "
        f"(Seed 42) — {task_name}",
        fontsize=11,
    )

    for name, history in histories.items():
        axes[0].plot(
            history["tr_loss"],
            label=name,
            color=COLORS[name],
            lw=1.8,
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for name, history in histories.items():
        axes[1].plot(
            [value * 100 for value in history["te_acc"]],
            label=name,
            color=COLORS[name],
            lw=1.8,
        )

    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT / f"compare_curves_{task_name}.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close()


def plot_attention_heatmaps(models_with_input, task_name):
    n_hops = 3
    n_models = len(models_with_input)

    fig, axes = plt.subplots(
        n_models,
        n_hops,
        figsize=(n_hops * 4, n_models * 3.5 + 1),
    )

    fig.suptitle(
        f"Standard vs PosDelta — Attention Maps "
        f"(Seed 42) — {task_name}",
        fontsize=11,
    )

    for row, (name, model, sample_ids, sample_mask, sample_words) in enumerate(
        models_with_input
    ):
        total_tokens = len(sample_words)
        words = sample_words[: min(total_tokens, 15)]
        tokens_to_show = min(total_tokens, 15)

        model.eval()

        with torch.no_grad():
            _ = model(sample_ids, sample_mask)

        for hop in range(n_hops):
            if n_models > 1:
                ax = axes[row][hop]
            else:
                ax = axes[hop]

            matrix = (
                model.attn_maps[hop][0, :tokens_to_show, :tokens_to_show]
                .cpu()
                .numpy()
            )

            sns.heatmap(
                matrix,
                ax=ax,
                xticklabels=words,
                yticklabels=words,
                cmap="Blues",
                vmin=0,
                vmax=matrix.max(),
                annot=(tokens_to_show <= 8),
                fmt=".2f",
                linewidths=0.3,
                linecolor="lightgray",
                cbar=(hop == n_hops - 1),
            )

            ax.set_title(f"{name} Hop {hop + 1}", fontsize=9)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.tight_layout()

    plt.savefig(
        OUT / f"compare_attn_{task_name}.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close()
