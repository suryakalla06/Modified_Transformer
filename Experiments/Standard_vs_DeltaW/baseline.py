from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    BABI_DIR,
    SEEDS,
    Config,
    count_params,
    init_weights,
    load_babi,
    make_loaders,
    pick_vocab_size,
    plot_attention_heatmaps,
    plot_training_curves,
    seed_everything,
    tokenize,
    Tokenizer,
    train_model,
)


OUT = Path("/kaggle/working/baseline_eval")


class WideAttentionBlock(nn.Module):
    def __init__(self, d, nh, d_attn, d_ff, drop):
        super().__init__()

        self.d = d
        self.nh = nh
        self.d_attn = d_attn
        self.dh = d_attn // nh

        self.WQ = nn.Linear(d, d_attn, bias=False)
        self.WK = nn.Linear(d, d_attn, bias=False)
        self.WV = nn.Linear(d, d_attn, bias=False)
        self.WO = nn.Linear(d_attn, d, bias=False)

        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

        self.ff = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, d),
        )

        self.dp = nn.Dropout(drop)

    def forward(self, x, pad_mask):
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
            .view(batch_size, seq_len, self.d_attn)
        )

        x = x + self.dp(self.WO(output))
        x = x + self.dp(self.ff(self.n2(x)))

        return x, attention.mean(1)


class StandardWideBaseline(nn.Module):
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

        # Match the HyperNet parameter budget by widening attention.
        v_dim = cfg.n_heads * cfg.d_model * cfg.rank
        u_dim = cfg.n_heads * (cfg.d_model // cfg.n_heads) * cfg.rank
        out_dim = 2 * (v_dim + u_dim)

        hyper_params = (
            (cfg.d_model * cfg.hyper_hidden + cfg.hyper_hidden)
            + (cfg.hyper_hidden * cfg.hyper_hidden + cfg.hyper_hidden)
            + (cfg.hyper_hidden * out_dim + out_dim)
            + 1
        )

        extra_params_per_hop = hyper_params // cfg.n_hops
        extra_d = extra_params_per_hop // (4 * cfg.d_model)

        d_attn = cfg.d_model + extra_d

        if d_attn % cfg.n_heads != 0:
            d_attn += cfg.n_heads - (d_attn % cfg.n_heads)

        self.d_attn = d_attn

        self.blks = nn.ModuleList(
            [
                WideAttentionBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    self.d_attn,
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


def main():
    cfg = Config()
    datasets = load_babi(BABI_DIR)

    print("\n" + "=" * 65)
    print(" BASELINE (WIDE ATTENTION) STRICT EVALUATION ")
    print("=" * 65)

    for task_name, splits in datasets.items():
        train_samples = splits["train"]
        test_samples = splits["test"]

        labels = sorted(
            set(sample["label"] for sample in train_samples + test_samples)
        )
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        cfg.num_classes = len(labels)

        seed_results = []

        print(f"\n[{task_name}] Starting Multi-Seed...")

        for seed in SEEDS:
            seed_everything(seed)

            vocab_cap = pick_vocab_size(
                len(train_samples),
                task_name,
            )

            tokenizer = Tokenizer()
            tokenizer.build(
                [s["text"] for s in train_samples + test_samples],
                maxv=vocab_cap,
            )
            cfg.vocab_size = tokenizer.vocab_size

            train_loader, test_loader = make_loaders(
                cfg,
                train_samples,
                test_samples,
                label_to_id,
                tokenizer,
                seed,
            )

            model = StandardWideBaseline(cfg).to(cfg.device)

            if seed == 42:
                print(
                    f"  -> Params: {count_params(model):,} | "
                    f"d_model: {cfg.d_model} | "
                    f"d_attn: {model.d_attn} | "
                    f"d_ff: {cfg.d_ff}"
                )

            print(f"  --- RUNNING SEED {seed} ---")

            accuracy, history = train_model(
                model,
                train_loader,
                test_loader,
                cfg,
                "Standard",
            )
            seed_results.append(accuracy * 100)

            if seed == 42:
                plot_training_curves(
                    {"Standard": history},
                    task_name,
                    OUT,
                    "Baseline",
                )

                sample = test_samples[0]
                sample_ids, sample_mask = tokenizer.encode(
                    sample["text"],
                    cfg.max_len,
                )

                sample_ids = sample_ids.unsqueeze(0).to(cfg.device)
                sample_mask = sample_mask.unsqueeze(0).to(cfg.device)

                words = tokenize(sample["text"])

                plot_attention_heatmaps(
                    {"Standard": model},
                    sample_ids,
                    sample_mask,
                    ["[CLS]"] + words,
                    task_name,
                    OUT,
                    "Baseline",
                )

        print(
            f"\n>>> [{task_name}] FINAL BASELINE: "
            f"{sum(seed_results) / len(seed_results):.2f}% ± "
            f"{float(torch.tensor(seed_results).std(unbiased=False)):.2f}%\n"
        )


if __name__ == "__main__":
    main()
