from pathlib import Path

import torch
import torch.nn as nn

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
    plot_dw_magnitude,
    plot_training_curves,
    seed_everything,
    tokenize,
    Tokenizer,
    train_model,
)
from common import HyperNet, DynamicAttentionBlock


OUT = Path("/kaggle/working/cls_delta_w_eval")


class CLSDeltaW(nn.Module):
    """
    Generate low-rank ΔW updates from the [CLS] representation
    of the current sequence at every hop.
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

        self.blks = nn.ModuleList(
            [
                DynamicAttentionBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_ff,
                    cfg.dropout,
                )
                for _ in range(cfg.n_hops)
            ]
        )

        self.hyper = HyperNet(
            cfg.d_model,
            cfg.n_heads,
            cfg.rank,
            cfg.hyper_hidden,
        )

        self.dw_mags = []
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

        self.dw_mags = []
        self.attn_maps = []

        for block in self.blks:
            deltas = self.hyper(x[:, 0, :])

            dQ = (deltas["Q"][0], deltas["Q"][1])
            dK = (deltas["K"][0], deltas["K"][1])

            if self.training:
                delta_q = torch.matmul(
                    dQ[0],
                    dQ[1].transpose(-1, -2),
                )
                self.dw_mags.append(
                    delta_q.norm(
                        p="fro",
                        dim=(-2, -1),
                    ).mean()
                )

            x, attention = block(
                x,
                pad_mask,
                dQ=dQ,
                dK=dK,
            )

            if not self.training:
                self.attn_maps.append(attention.detach())

        return self.clf(x[:, 0, :])


def main():
    cfg = Config()
    datasets = load_babi(BABI_DIR)

    print("\n" + "=" * 65)
    print(" HYPERNET: CLS → ΔW STRICT EVALUATION ")
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

            model = CLSDeltaW(cfg).to(cfg.device)

            if seed == 42:
                print(f"  -> Params: {count_params(model):,}")
                print(
                    f"     d_model: {cfg.d_model} | "
                    f"d_ff: {cfg.d_ff}"
                )

            print(f"\n  --- RUNNING SEED {seed} ---")

            accuracy, history = train_model(
                model,
                train_loader,
                test_loader,
                cfg,
                "CLS",
                track_dw=True,
            )
            seed_results.append(accuracy * 100)

            if seed == 42:
                plot_training_curves(
                    {"CLS→ΔW": history},
                    task_name,
                    OUT,
                    "CLS→ΔW",
                )

                plot_dw_magnitude(
                    {"CLS→ΔW": history},
                    task_name,
                    OUT,
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
                    {"CLS→ΔW": model},
                    sample_ids,
                    sample_mask,
                    ["[CLS]"] + words,
                    task_name,
                    OUT,
                    "CLS→ΔW",
                )

        print(
            f"\n>>> [{task_name}] FINAL CLS→ΔW: "
            f"{sum(seed_results) / len(seed_results):.2f}% ± "
            f"{float(torch.tensor(seed_results).std(unbiased=False)):.2f}%\n"
        )


if __name__ == "__main__":
    main()
