import csv
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
    BABI_DIR,
    SEEDS,
    Config,
    build_loaders,
    build_pos_delta,
    count_params,
    init_weights,
    load_babi,
    masked_mean,
    plot_attention_heatmaps,
    plot_training_curves,
    pick_vocab_size,
    seed_everything,
    tokenize,
    train_model,
)


class StandardAttentionBlock(nn.Module):
    def __init__(self, d, num_heads, d_ff, dropout):
        super().__init__()

        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d, bias=False)

        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )

        self.dp = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        batch_size, seq_len, _ = x.shape

        n = self.n1(x)

        Q = self.WQ(n)
        K = self.WK(n)
        V = self.WV(n)

        def split_heads(t):
            return t.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        scores = (
            torch.matmul(Q, K.transpose(-2, -1))
            / math.sqrt(self.head_dim)
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


class StandardBaseline(nn.Module):
    """
    Standard baseline:
    CLS token + vanilla attention + d_attn = d_model.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.emb = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=0,
        )
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.nin = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.clf = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

        self.d_attn = cfg.d_model

        self.blks = nn.ModuleList(
            [
                StandardAttentionBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                )
                for _ in range(cfg.n_hops)
            ]
        )

        self.attn_maps = []

        init_weights(self)

    def forward(self, ids, mask):
        positions = torch.arange(
            ids.shape[1],
            device=ids.device,
        ).unsqueeze(0)

        x = self.drop(
            self.nin(self.emb(ids) + self.pos(positions))
        )

        pad_mask = (
            (mask == 0)
            .unsqueeze(1)
            .unsqueeze(2)
            .float()
            * -1e9
        )

        self.attn_maps = []

        for block in self.blks:
            x, attention = block(x, pad_mask)

            if not self.training:
                self.attn_maps.append(attention.detach())

        return self.clf(x[:, 0, :])


class PosDeltaAttentionBlock(nn.Module):
    """
    Standard attention with fixed rank-k positional ΔW on Q/K,
    controlled by two learnable scalar gates.
    """

    def __init__(
        self,
        d,
        num_heads,
        d_ff,
        dropout,
        max_len,
        init_scale,
        rank,
    ):
        super().__init__()

        self.d = d
        self.num_heads = num_heads
        self.head_dim = d // num_heads

        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d, bias=False)

        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d),
        )

        self.dp = nn.Dropout(dropout)

        dW_Q = build_pos_delta(
            max_len,
            d,
            d,
            rank,
            base_row=10000.0,
            base_col=7919.0,
        )

        dW_K = build_pos_delta(
            max_len,
            d,
            d,
            rank,
            base_row=9973.0,
            base_col=6151.0,
        )

        self.register_buffer(
            "dW_Q",
            dW_Q,
            persistent=False,
        )
        self.register_buffer(
            "dW_K",
            dW_K,
            persistent=False,
        )

        self.scale_Q = nn.Parameter(
            torch.tensor([float(init_scale)])
        )
        self.scale_K = nn.Parameter(
            torch.tensor([float(init_scale)])
        )

    def forward(self, x, pad_mask):
        batch_size, seq_len, dim = x.shape

        n = self.n1(x)

        Q_base = self.WQ(n)
        K_base = self.WK(n)
        V = self.WV(n)

        dQ = self.dW_Q[:seq_len]
        dK = self.dW_K[:seq_len]

        Q_pos = (
            torch.einsum("btd,tde->bte", n, dQ)
            * self.scale_Q
        )

        K_pos = (
            torch.einsum("btd,tde->bte", n, dK)
            * self.scale_K
        )

        Q = Q_base + Q_pos
        K = K_base + K_pos

        def split_heads(t):
            return t.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim,
            ).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        scores = (
            torch.matmul(Q, K.transpose(-2, -1))
            / math.sqrt(self.head_dim)
            + pad_mask
        )

        attention = self.dp(F.softmax(scores, dim=-1))

        output = (
            torch.matmul(attention, V)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, dim)
        )

        x = x + self.dp(self.WO(output))
        x = x + self.dp(self.ff(self.n2(x)))

        return x, attention.mean(1)


class PosDeltaBaseline(nn.Module):
    """
    PosDelta baseline:
    no CLS token + masked-mean readout + fixed rank-k
    positional ΔW on Q/K with learnable scalar gates.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.emb = nn.Embedding(
            cfg.vocab_size,
            cfg.d_model,
            padding_idx=0,
        )
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.nin = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.clf = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, cfg.num_classes),
        )

        self.d_attn = cfg.d_model

        self.blks = nn.ModuleList(
            [
                PosDeltaAttentionBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                    cfg.max_len,
                    cfg.pos_delta_init,
                    cfg.pos_delta_rank,
                )
                for _ in range(cfg.n_hops)
            ]
        )

        self.attn_maps = []

        init_weights(self)

    def forward(self, ids, mask):
        positions = torch.arange(
            ids.shape[1],
            device=ids.device,
        ).unsqueeze(0)

        x = self.drop(
            self.nin(self.emb(ids) + self.pos(positions))
        )

        pad_mask = (
            (mask == 0)
            .unsqueeze(1)
            .unsqueeze(2)
            .float()
            * -1e9
        )

        self.attn_maps = []

        for block in self.blks:
            x, attention = block(x, pad_mask)

            if not self.training:
                self.attn_maps.append(attention.detach())

        return self.clf(masked_mean(x, mask))


def main():
    cfg = Config()
    datasets = load_babi(BABI_DIR)

    print("\n" + "=" * 80)
    print(
        " STANDARD vs POSDELTA "
        "(parameter-equalized, learnable scale) — STRICT EVAL "
    )
    print("=" * 80)

    results = {}
    per_seed_log = []

    for task_name, splits in datasets.items():
        train_samples = splits["train"]
        test_samples = splits["test"]

        labels = sorted(
            set(
                sample["label"]
                for sample in train_samples + test_samples
            )
        )
        label_to_id = {
            label: index for index, label in enumerate(labels)
        }

        cfg.num_classes = len(labels)

        results[task_name] = {
            "Standard": [],
            "PosDelta": [],
        }

        seed42_histories = {}
        seed42_attention = []

        print(f"\n[{task_name}]")

        for seed in SEEDS:
            print(f"  -- Seed {seed} --")

            for variant, model_cls in [
                ("Standard", StandardBaseline),
                ("PosDelta", PosDeltaBaseline),
            ]:
                seed_everything(seed)

                vocab_cap = pick_vocab_size(
                    len(train_samples),
                    task_name,
                )

                tokenizer, train_loader, test_loader = build_loaders(
                    variant,
                    train_samples,
                    test_samples,
                    label_to_id,
                    vocab_cap,
                    cfg,
                    seed,
                )

                model = model_cls(cfg).to(cfg.device)

                if seed == 42:
                    if variant == "Standard":
                        extra = f"d_attn={model.d_attn}"
                    else:
                        extra = (
                            f"d_attn={model.d_attn} "
                            f"init_scale={cfg.pos_delta_init} "
                            f"rank={cfg.pos_delta_rank}"
                        )

                    print(
                        f"    [{variant}] "
                        f"params={count_params(model):,}  {extra}"
                    )

                accuracy, history = train_model(
                    model,
                    train_loader,
                    test_loader,
                    cfg,
                    variant,
                )

                results[task_name][variant].append(
                    accuracy * 100
                )

                per_seed_log.append(
                    (
                        task_name,
                        variant,
                        seed,
                        accuracy * 100,
                    )
                )

                print(
                    f"    [{variant}] "
                    f"seed {seed} → {accuracy * 100:.2f}%"
                )

                if variant == "PosDelta":
                    scale_readout = []

                    for hop, block in enumerate(model.blks):
                        q_scale = block.scale_Q.item()
                        k_scale = block.scale_K.item()

                        scale_readout.append(
                            f"H{hop + 1}"
                            f"[Q:{q_scale:+.3f}, "
                            f"K:{k_scale:+.3f}]"
                        )

                    print(
                        "      Learned Scales: "
                        + " | ".join(scale_readout)
                    )

                if seed == 42:
                    seed42_histories[variant] = history

                    sample = test_samples[0]

                    sample_ids, sample_mask = tokenizer.encode(
                        sample["text"],
                        cfg.max_len,
                    )

                    sample_ids = (
                        sample_ids.unsqueeze(0)
                        .to(cfg.device)
                    )
                    sample_mask = (
                        sample_mask.unsqueeze(0)
                        .to(cfg.device)
                    )

                    words = tokenize(sample["text"])

                    if variant == "Standard":
                        seed42_attention.append(
                            (
                                variant,
                                model,
                                sample_ids,
                                sample_mask,
                                ["[CLS]"] + words,
                            )
                        )
                    else:
                        seed42_attention.append(
                            (
                                variant,
                                model,
                                sample_ids,
                                sample_mask,
                                words,
                            )
                        )

        plot_training_curves(
            seed42_histories,
            task_name,
        )

        plot_attention_heatmaps(
            seed42_attention,
            task_name,
        )

        standard = np.asarray(
            results[task_name]["Standard"]
        )
        posdelta = np.asarray(
            results[task_name]["PosDelta"]
        )

        delta = (
            posdelta.mean()
            - standard.mean()
        )

        print(
            f"\n  >>> [{task_name}] "
            f"Standard: {standard.mean():.2f}% ± {standard.std():.2f}%   "
            f"PosDelta: {posdelta.mean():.2f}% ± {posdelta.std():.2f}%   "
            f"Δ: {delta:+.2f}%"
        )

    print("\n" + "=" * 88)
    print(" FINAL COMPARISON ".center(88, "="))
    print("=" * 88)

    header = (
        f"{'Task':<14} | "
        f"{'Standard (mean ± std)':<24} | "
        f"{'PosDelta (mean ± std)':<24} | "
        f"{'Δ':>8}"
    )
    print(header)
    print("-" * 88)

    for task_name in results:
        standard = np.asarray(
            results[task_name]["Standard"]
        )
        posdelta = np.asarray(
            results[task_name]["PosDelta"]
        )

        delta = (
            posdelta.mean()
            - standard.mean()
        )

        row = (
            f"{task_name:<14} | "
            f"{standard.mean():6.2f}% ± {standard.std():4.2f}%        | "
            f"{posdelta.mean():6.2f}% ± {posdelta.std():4.2f}%        | "
            f"{delta:+7.2f}%"
        )

        print(row)

    print("=" * 88)

    per_seed_path = OUT / "comparison_results.csv"

    with open(per_seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["task", "variant", "seed", "accuracy"]
        )
        writer.writerows(per_seed_log)

    summary_path = OUT / "comparison_summary.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "task",
                "standard_mean",
                "standard_std",
                "posdelta_mean",
                "posdelta_std",
                "delta",
            ]
        )

        for task_name in results:
            standard = np.asarray(
                results[task_name]["Standard"]
            )
            posdelta = np.asarray(
                results[task_name]["PosDelta"]
            )

            writer.writerow(
                [
                    task_name,
                    f"{standard.mean():.4f}",
                    f"{standard.std():.4f}",
                    f"{posdelta.mean():.4f}",
                    f"{posdelta.std():.4f}",
                    f"{posdelta.mean() - standard.mean():.4f}",
                ]
            )

    print(f"\nPer-seed results saved to: {per_seed_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
