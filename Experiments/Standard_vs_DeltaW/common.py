import os
import re
import math
import random
from copy import deepcopy
from collections import Counter
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


# ---------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------

class Config:
    vocab_size = 500
    d_model = 128
    n_heads = 2
    n_hops = 3
    d_ff = 256
    max_len = 200
    dropout = 0.1
    label_smooth = 0.1
    rank = 8
    hyper_hidden = 32
    num_classes = None
    batch_size = 64
    epochs = 60
    patience = 20
    lr = 5e-4
    weight_decay = 1e-2
    device = "cuda" if torch.cuda.is_available() else "cpu"


BABI_DIR = (
    "/kaggle/input/datasets/roblexnana/"
    "the-babi-tasks-for-nlp-qa-system/tasks_1-20_v1-2/en-10k"
)

SEEDS = [42, 43, 44, 45, 46]


# ---------------------------------------------------------------------
# Reproducibility / utilities
# ---------------------------------------------------------------------

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


def init_weights(module):
    for mod in module.modules():
        if isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seq_mean(x, mask):
    """Mean over real tokens only; padded tokens are ignored."""
    return (
        (x * mask.unsqueeze(-1)).sum(1)
        / mask.sum(1, keepdim=True).float().clamp(min=1)
    )


# ---------------------------------------------------------------------
# Tokenization / dataset
# ---------------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b[\w']+\b", text.lower())


class Tokenizer:
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
                self.label_to_id[sample["label"]], dtype=torch.long
            ),
        }


def pick_vocab_size(n_train_samples, task_name):
    if "bAbI" in task_name or "qa" in task_name.lower():
        return 200
    if n_train_samples < 5000:
        return 5000
    return 30000


# ---------------------------------------------------------------------
# bAbI loading
# ---------------------------------------------------------------------

def parse_babi(path):
    samples = []

    with open(path) as f:
        ctx = {}

        for line in f:
            line = line.strip()
            if not line:
                continue

            line_id, text = line.split(" ", 1)
            line_id = int(line_id)

            if line_id == 1:
                ctx = {}

            if "\t" in text:
                question, answer, _ = text.split("\t")
                samples.append(
                    {
                        "text": " ".join(ctx[k] for k in sorted(ctx))
                        + " "
                        + question.strip(),
                        "label": answer.strip(),
                    }
                )
            else:
                ctx[line_id] = text.strip()

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


def make_loaders(cfg, train_samples, test_samples, label_to_id, tokenizer, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }

    train_loader = DataLoader(
        TextDataset(train_samples, label_to_id, tokenizer, cfg.max_len),
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        TextDataset(test_samples, label_to_id, tokenizer, cfg.max_len),
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------

def train_model(model, train_loader, test_loader, cfg, name, track_dw=False):
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
        "dw_mags": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        epoch_dw_mag = 0.0

        for batch in train_loader:
            ids = batch["input_ids"].to(cfg.device)
            mask = batch["attention_mask"].to(cfg.device)
            labels = batch["label"].to(cfg.device)

            optimizer.zero_grad()

            logits = model(ids, mask)
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=cfg.label_smooth,
            )

            if track_dw and getattr(model, "dw_mags", None):
                avg_batch_dw = sum(model.dw_mags) / len(model.dw_mags)
                epoch_dw_mag += avg_batch_dw.item()

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
            np.mean(np.asarray(predictions) == np.asarray(labels))
        )

        avg_loss = total_loss / batches
        avg_dw = (
            epoch_dw_mag / batches
            if track_dw and batches > 0
            else 0.0
        )

        history["tr_loss"].append(avg_loss)
        history["te_acc"].append(test_acc)

        if track_dw and avg_dw > 0:
            history["dw_mags"].append(avg_dw)

        dw_str = f" | ΔW: {avg_dw:.4f}" if track_dw and avg_dw > 0 else ""
        print(
            f"    [{name}] ep{epoch:2d} "
            f"loss: {avg_loss:.4f} | "
            f"val_acc: {test_acc * 100:.1f}%{dw_str}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= cfg.patience:
            print(f"    [Early Stop at ep {epoch}]")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_acc, history


# ---------------------------------------------------------------------
# Shared plotting
# ---------------------------------------------------------------------

def plot_training_curves(histories, task_name, output_dir, title):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{title} (Seed 42) — {task_name}", fontsize=11)

    for name, history in histories.items():
        axes[0].plot(history["tr_loss"], label=name, lw=1.8)

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for name, history in histories.items():
        axes[1].plot(
            [value * 100 for value in history["te_acc"]],
            label=name,
            lw=1.8,
        )

    axes[1].set_title("Test Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"curves_{task_name}.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close()


def plot_dw_magnitude(histories, task_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))

    for name, history in histories.items():
        if history.get("dw_mags"):
            ax.plot(
                history["dw_mags"],
                label=name,
                lw=1.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg ΔW Norm")
    ax.set_title(f"ΔW Magnitude (Seed 42) — {task_name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"dw_magnitude_{task_name}.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close()


def plot_attention_heatmaps(
    models,
    sample_ids,
    sample_mask,
    sample_words,
    task_name,
    output_dir,
    title,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_hops = 3
    n_models = len(models)
    total_tokens = len(sample_words)
    words = sample_words[: min(total_tokens, 15)]
    tokens_to_show = min(total_tokens, 15)

    fig, axes = plt.subplots(
        n_models,
        n_hops,
        figsize=(n_hops * 4, n_models * 3.5 + 1),
    )
    fig.suptitle(f"{title} Attention Maps (Seed 42) — {task_name}", fontsize=11)

    for row, (name, model) in enumerate(models.items()):
        model.eval()

        with torch.no_grad():
            _ = model(sample_ids, sample_mask)

        for hop in range(n_hops):
            if n_models > 1:
                ax = axes[row][hop]
            elif n_hops > 1:
                ax = axes[hop]
            else:
                ax = axes

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
        output_dir / f"attention_{task_name}.png",
        dpi=130,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------
# Shared HyperNet components
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNet(nn.Module):
    def __init__(self, d, nh, rank, hidden):
        super().__init__()

        self.d = d
        self.nh = nh
        self.r = rank
        self.dh = d // nh

        self.v_dim = nh * d * rank
        self.u_dim = nh * self.dh * rank

        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(
                hidden,
                2 * (self.v_dim + self.u_dim),
            ),
        )

        self.scale = nn.Parameter(torch.tensor(1.0))

    def _split_uv(self, raw_mat):
        V = raw_mat[:, : self.v_dim]
        U = raw_mat[:, self.v_dim:]

        return (
            U.view(-1, self.nh, self.dh, self.r),
            V.view(-1, self.nh, self.d, self.r),
        )

    def forward(self, v):
        raw = self.net(v) * self.scale.abs()
        raw_Q, raw_K = raw.chunk(2, dim=-1)

        return {
            "Q": self._split_uv(raw_Q),
            "K": self._split_uv(raw_K),
        }


class DynamicAttentionBlock(nn.Module):
    def __init__(self, d, nh, d_ff, drop):
        super().__init__()

        self.d = d
        self.nh = nh
        self.dh = d // nh

        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d, bias=False)

        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, d),
        )

        self.dp = nn.Dropout(drop)

    def forward(self, x, pad_mask, dQ=None, dK=None):
        batch_size, seq_len, _ = x.shape

        n = self.n1(x)
        Q, K, V = self.WQ(n), self.WK(n), self.WV(n)

        def split_heads(t):
            return t.view(
                batch_size,
                seq_len,
                self.nh,
                self.dh,
            ).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        n_expanded = n.unsqueeze(1)

        if dQ is not None:
            Q = Q + torch.matmul(
                torch.matmul(n_expanded, dQ[1]),
                dQ[0].transpose(-1, -2),
            )

        if dK is not None:
            K = K + torch.matmul(
                torch.matmul(n_expanded, dK[1]),
                dK[0].transpose(-1, -2),
            )

        scores = (
            torch.matmul(Q, K.transpose(-2, -1))
            / (self.dh ** 0.5)
            + pad_mask
        )

        attention = self.dp(F.softmax(scores, dim=-1))

        output = (
            torch.matmul(attention, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d)
        )

        x = x + self.dp(self.WO(output))
        x = x + self.dp(self.ff(self.n2(x)))

        return x, attention.mean(1)
