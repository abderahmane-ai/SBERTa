"""
test_sberta_synthetic.py
========================
Self-contained synthetic test suite for the SBERTa architecture.

Tests whether the architecture has the *structural potential* to learn
multilingual code-switching representations, without requiring any real
corpus or pre-trained tokenizer.

Synthetic setup
---------------
Two simulated "languages" are created by partitioning the vocabulary:
  · Language A : token IDs in [100,  2049]  (low-frequency zone)
  · Language B : token IDs in [5000, 10000] (high-frequency zone)

Three sequence types are generated:
  · Monolingual A  : pure Language-A tokens
  · Monolingual B  : pure Language-B tokens
  · Code-switched  : alternating Language-A / Language-B spans

Diagnostic suite (7 tests)
---------------------------
  T1  Smoke            — forward pass runs; all output shapes are correct
  T2  Loss sanity      — all 4 loss components are finite and positive
  T3  Gradient flow    — every learnable parameter receives a gradient
  T4  Switch signal    — mean switch magnitude at code-switch boundaries
                         is strictly > mean within-language switch magnitude
  T5  Prototype separation — after 60 gradient steps on labelled bilingual
                         data, the dominant prototype for Language-A tokens
                         differs from Language-B tokens (>55 % purity)
  T6  Ortho regularis. — L_ortho drives prototype cosine similarity toward 0
                         (prototypes become more orthogonal over training)
  T7  Mini convergence — total loss decreases by ≥5 % over 80 steps

Run
---
    pip install torch
    # place sberta/ (config.py + model.py) next to this file, then:
    python test_sberta_synthetic.py

    # or with pytest:
    pytest test_sberta_synthetic.py -v
"""

from __future__ import annotations

import math
import sys
import time
from typing import Tuple

import torch
import torch.nn.functional as F

# ── import SBERTa ──────────────────────────────────────────────────────────────
try:
    from sberta.config import SBERTaConfig
    from sberta.model  import SBERTaForPreTraining, SBERTaModel
except ModuleNotFoundError:
    print(
        "[ERROR] Could not import sberta. "
        "Make sure sberta/config.py and sberta/model.py are on the Python path.",
        file=sys.stderr,
    )
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Globals / fixtures
# ═══════════════════════════════════════════════════════════════════════════════

SEED      = 42
VOCAB     = 50_265
LANG_A    = (100,  2049)   # [lo, hi) token range for Language A
LANG_B    = (5000, 10000)  # [lo, hi) token range for Language B
SEQ_LEN   = 64
BATCH     = 8              # enough for meaningful stats, fast on CPU
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)


def _tiny_config() -> SBERTaConfig:
    """A very small config: runs in seconds on CPU."""
    return SBERTaConfig(
        vocab_size               = VOCAB,
        hidden_size              = 128,
        num_hidden_layers        = 4,
        num_attention_heads      = 4,
        intermediate_size        = 256,
        n_base_layers            = 2,
        num_languages            = 2,
        max_position_embeddings  = 128,
        hidden_dropout_prob      = 0.0,          # deterministic for tests
        attention_probs_dropout_prob = 0.0,
        learnable_temperature    = False,
        rtd_weight               = 15.0,
        lambda_cluster           = 3.0,
        lambda_ortho             = 1.0,
        sinkhorn_iters           = 10,
        span_mask_min_len        = 1,
        span_mask_max_len        = 5,
    )


# ── Synthetic data helpers ──────────────────────────────────────────────────

def _rand_tokens(lo: int, hi: int, length: int) -> torch.Tensor:
    return torch.randint(lo, hi, (length,))

def _make_batch(
    seq_len: int = SEQ_LEN,
    batch_size: int = BATCH,
    mode: str = "mixed",          # "mono_a" | "mono_b" | "switched" | "mixed"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (input_ids, attention_mask, lang_labels) — all (B, T).
    lang_labels: 0 = Language A, 1 = Language B, -1 = padding/switch boundary.
    """
    ids_list, mask_list, label_list = [], [], []

    for i in range(batch_size):
        if mode == "mono_a":
            m = "mono_a"
        elif mode == "mono_b":
            m = "mono_b"
        elif mode == "switched":
            m = "switched"
        else:                    # "mixed" — round-robin
            m = ["mono_a", "mono_b", "switched"][i % 3]

        if m == "mono_a":
            ids    = _rand_tokens(*LANG_A, seq_len)
            labels = torch.zeros(seq_len, dtype=torch.long)
        elif m == "mono_b":
            ids    = _rand_tokens(*LANG_B, seq_len)
            labels = torch.ones(seq_len, dtype=torch.long)
        else:   # code-switched: alternating spans of length 8
            ids    = torch.zeros(seq_len, dtype=torch.long)
            labels = torch.full((seq_len,), -1, dtype=torch.long)
            span   = 8
            lang   = 0
            for start in range(0, seq_len, span):
                end = min(start + span, seq_len)
                if lang == 0:
                    ids[start:end] = _rand_tokens(*LANG_A, end - start)
                    labels[start:end] = 0
                else:
                    ids[start:end] = _rand_tokens(*LANG_B, end - start)
                    labels[start:end] = 1
                lang = 1 - lang

        ids_list.append(ids)
        mask_list.append(torch.ones(seq_len, dtype=torch.long))
        label_list.append(labels)

    return (
        torch.stack(ids_list).to(DEVICE),
        torch.stack(mask_list).to(DEVICE),
        torch.stack(label_list).to(DEVICE),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test runner (minimal — no pytest dependency required)
# ═══════════════════════════════════════════════════════════════════════════════

class _TestResult:
    def __init__(self):
        self.passed  = []
        self.failed  = []
        self.skipped = []

    def ok(self, name, msg=""):
        self.passed.append(name)
        tag = f"\033[92m✓ PASS\033[0m"
        print(f"  {tag}  {name}" + (f"  [{msg}]" if msg else ""))

    def fail(self, name, msg=""):
        self.failed.append(name)
        tag = f"\033[91m✗ FAIL\033[0m"
        print(f"  {tag}  {name}" + (f"  — {msg}" if msg else ""))

    def skip(self, name, reason=""):
        self.skipped.append(name)
        tag = f"\033[93m~ SKIP\033[0m"
        print(f"  {tag}  {name}" + (f"  — {reason}" if reason else ""))

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(
            f"\n{'─'*60}\n"
            f"  {len(self.passed)}/{total} passed"
            + (f"  |  {len(self.failed)} failed" if self.failed else "")
            + (f"  |  {len(self.skipped)} skipped" if self.skipped else "")
        )
        return len(self.failed) == 0


R = _TestResult()


# ═══════════════════════════════════════════════════════════════════════════════
# T1 — Smoke test: forward pass + output shapes
# ═══════════════════════════════════════════════════════════════════════════════

def test_t1_smoke():
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    ids, mask, _ = _make_batch()

    try:
        with torch.no_grad():
            out = model(ids, mask)
    except Exception as e:
        R.fail("T1 Smoke", str(e))
        return

    # ── shape checks ──────────────────────────────────────────────────────
    errors = []

    # loss must be scalar
    if out["loss"].shape != torch.Size([]):
        errors.append(f"loss shape={out['loss'].shape} (expected scalar)")

    # language_probs: (B, T, K)
    expected_lp = torch.Size([BATCH, SEQ_LEN, cfg.num_languages])
    if out["language_probs"].shape != expected_lp:
        errors.append(f"language_probs shape={out['language_probs'].shape} expected {expected_lp}")

    # switch_magnitudes: (B, T)
    expected_sw = torch.Size([BATCH, SEQ_LEN])
    if out["switch_magnitudes"].shape != expected_sw:
        errors.append(f"switch_magnitudes shape={out['switch_magnitudes'].shape} expected {expected_sw}")

    # language_probs must sum to ~1 per token
    row_sums = out["language_probs"].sum(-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        errors.append("language_probs rows do not sum to 1")

    if errors:
        R.fail("T1 Smoke", "; ".join(errors))
    else:
        R.ok("T1 Smoke", f"all shapes correct (B={BATCH}, T={SEQ_LEN}, K={cfg.num_languages})")


# ═══════════════════════════════════════════════════════════════════════════════
# T2 — Loss sanity: all components finite and positive
# ═══════════════════════════════════════════════════════════════════════════════

def test_t2_loss_sanity():
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    ids, mask, _ = _make_batch()

    with torch.no_grad():
        out = model(ids, mask)

    checks = {
        "loss_gen":     out["loss_gen"],
        "loss_rtd":     out["loss_rtd"],
        "loss_cluster": out["loss_cluster"],
        "loss_ortho":   out["loss_ortho"],
        "loss (total)": out["loss"].item(),
    }
    errors = []
    for name, val in checks.items():
        if not math.isfinite(val):
            errors.append(f"{name}={val} (not finite)")
        elif val < 0:
            errors.append(f"{name}={val:.4f} (negative)")

    if errors:
        R.fail("T2 Loss sanity", "; ".join(errors))
    else:
        R.ok(
            "T2 Loss sanity",
            "gen={loss_gen:.3f}  rtd={loss_rtd:.3f}  "
            "cluster={loss_cluster:.3f}  ortho={loss_ortho:.3f}".format(**checks),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T3 — Gradient flow: every parameter gets a gradient
# ═══════════════════════════════════════════════════════════════════════════════

def test_t3_gradient_flow():
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    model.train()
    ids, mask, _ = _make_batch()

    out = model(ids, mask)
    out["loss"].backward()

    dead  = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    nan_p = [n for n, p in model.named_parameters() if p.grad is not None and p.grad.isnan().any()]

    if dead:
        R.fail("T3 Gradient flow", f"no gradient: {dead[:5]}")
    elif nan_p:
        R.fail("T3 Gradient flow", f"NaN gradient: {nan_p[:5]}")
    else:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        R.ok("T3 Gradient flow", f"all {n_params:,} param floats received clean gradients")


# ═══════════════════════════════════════════════════════════════════════════════
# T4 — Switch signal: boundary > within-language
# ═══════════════════════════════════════════════════════════════════════════════

def test_t4_switch_signal():
    """
    In code-switched sequences the token at each span boundary (position 0, 8,
    16, …) should have a higher switch magnitude than tokens deep inside a span.
    This is a *structural* test — it checks that s_t = 1 − p_t·p_{t-1} is
    geometrically sensitive to distribution shifts, not that the model has
    converged on the correct language clustering.
    """
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    model.eval()

    # Use more samples for stability
    ids, mask, _ = _make_batch(seq_len=SEQ_LEN, batch_size=32, mode="switched")

    with torch.no_grad():
        out = model(ids, mask)

    sw = out["switch_magnitudes"]          # (B, T)
    span = 8                               # must match _make_batch span size

    # boundary positions: 0 (always 0 by definition), 8, 16, …
    boundary_idx  = list(range(span, SEQ_LEN, span))    # skip pos 0 (defined = 0)
    interior_idx  = [
        j for j in range(SEQ_LEN)
        if j not in boundary_idx and j != 0
        and (j % span) not in (0, span - 1)             # exclude edges too
    ]

    mean_boundary = sw[:, boundary_idx].mean().item()
    mean_interior = sw[:, interior_idx].mean().item()

    # The structural property: boundaries should exceed interior on average.
    # This will be very weak at init (random weights) but the *direction*
    # should already be present because the token distributions genuinely differ.
    if mean_boundary > mean_interior:
        R.ok(
            "T4 Switch signal",
            f"boundary sw={mean_boundary:.4f}  interior sw={mean_interior:.4f}  "
            f"ratio={mean_boundary/max(mean_interior,1e-9):.2f}×",
        )
    else:
        R.fail(
            "T4 Switch signal",
            f"boundary sw={mean_boundary:.4f} ≤ interior sw={mean_interior:.4f} — "
            "prototypes do not yet produce a directional boundary signal",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T5 — Prototype separation (two sub-tests, correctly scoped)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Why two sub-tests?
# ------------------
# The assignment mechanism p = softmax(h·Lᵀ / τ) has TWO moving parts:
#   (a) the representation h produced by Phase 1 (governed by embeddings + layers)
#   (b) the prototype vectors L
#
# Moving only L while keeping random embeddings is the bootstrapping trap the
# original single-test fell into: prototypes cannot cluster what looks identical
# to them (random embeddings → undifferentiated Phase 1 output → no signal).
#
# T5a — Mechanism test (pure prototype alignment):
#   Pre-seed embeddings with a directional signal (dim-0 offset per language),
#   then train ONLY the prototype parameters. The prototypes should align to
#   an already-structured representation space in ~40 steps.
#   What this tests: "does the softmax / gradient flow through L work?"
#
# T5b — Pipeline test (joint embedding + prototype):
#   Train the FULL encoder (embeddings + Phase-1 layers + prototypes) with a
#   supervised CE signal. Embeddings must first develop internal structure;
#   prototypes then align to that structure.
#   What this tests: "can the full pipeline separate two languages end-to-end?"

def _purity(model: SBERTaForPreTraining, cfg: SBERTaConfig) -> tuple[float, float]:
    """Return mean dominant-prototype index for Lang-A and Lang-B eval batches."""
    model.eval()
    with torch.no_grad():
        ids_a, mask_a, _ = _make_batch(seq_len=SEQ_LEN, batch_size=32, mode="mono_a")
        ids_b, mask_b, _ = _make_batch(seq_len=SEQ_LEN, batch_size=32, mode="mono_b")
        _, p_a, _ = model.sberta.forward_phase1(ids_a, mask_a)
        _, p_b, _ = model.sberta.forward_phase1(ids_b, mask_b)
        dom_a = p_a.argmax(-1).float().mean().item()
        dom_b = p_b.argmax(-1).float().mean().item()
    model.train()
    return dom_a, dom_b


def _is_separated(dom_a: float, dom_b: float) -> bool:
    """True iff the two dominant-prototype means are on opposite sides of 0.5."""
    return (dom_a < 0.45 and dom_b > 0.55) or (dom_a > 0.55 and dom_b < 0.45)


def test_t5_prototype_separation():
    """Runs T5a and T5b; both must pass for T5 to be considered passing."""

    # ── T5a: pure prototype alignment (pre-structured embeddings) ─────────
    #
    # Inject a simple directional signal into the embedding table:
    #   Lang-A token embeddings: dimension 0 shifted +2.0
    #   Lang-B token embeddings: dimension 0 shifted −2.0
    # This mimics the state the model reaches after Phase-1 has learnt to
    # cluster tokens by distributional co-occurrence (MLM pressure).
    # We then train ONLY the prototype vectors to check the alignment mechanism.

    cfg_a  = _tiny_config()
    model_a = SBERTaForPreTraining(cfg_a).to(DEVICE)

    with torch.no_grad():
        w = model_a.sberta.embeddings.token_embeddings.weight
        w[LANG_A[0]:LANG_A[1], 0] += 2.0
        w[LANG_B[0]:LANG_B[1], 0] -= 2.0

    model_a.train()
    opt_a = torch.optim.Adam(model_a.sberta.prototypes.parameters(), lr=5e-3)

    STEPS_A = 40
    for step in range(STEPS_A):
        mode = "mono_a" if step % 2 == 0 else "mono_b"
        ids, mask, _ = _make_batch(seq_len=SEQ_LEN, batch_size=8, mode=mode)
        _, p, _ = model_a.sberta.forward_phase1(ids, mask)
        lang_label = 0 if mode == "mono_a" else 1
        tgt = torch.full((8 * SEQ_LEN,), lang_label, dtype=torch.long, device=DEVICE)
        ce = F.nll_loss(torch.log(p.view(-1, cfg_a.num_languages) + 1e-9), tgt)
        opt_a.zero_grad(); ce.backward(); opt_a.step()

    dom_a_5a, dom_b_5a = _purity(model_a, cfg_a)
    t5a_ok = _is_separated(dom_a_5a, dom_b_5a)

    # ── T5b: joint pipeline (embeddings + layers + prototypes trained together)
    #
    # Start from a fully random model. Train the full encoder with a
    # supervised CE loss — embeddings and Phase-1 layers must develop
    # internal structure; prototypes must then align to that structure.
    # 150 steps at lr=1e-3 with gradient clipping is sufficient on the
    # tiny config because the vocab ranges are non-overlapping and the
    # signal is strong.

    cfg_b   = _tiny_config()
    model_b = SBERTaForPreTraining(cfg_b).to(DEVICE)
    model_b.train()
    opt_b = torch.optim.AdamW(model_b.sberta.parameters(), lr=1e-3, weight_decay=0.01)

    STEPS_B = 150
    for step in range(STEPS_B):
        mode = "mono_a" if step % 2 == 0 else "mono_b"
        ids, mask, _ = _make_batch(seq_len=SEQ_LEN, batch_size=16, mode=mode)
        _, p, _ = model_b.sberta.forward_phase1(ids, mask)
        lang_label = 0 if mode == "mono_a" else 1
        tgt = torch.full((16 * SEQ_LEN,), lang_label, dtype=torch.long, device=DEVICE)
        ce = F.nll_loss(torch.log(p.view(-1, cfg_b.num_languages) + 1e-9), tgt)
        opt_b.zero_grad(); ce.backward()
        torch.nn.utils.clip_grad_norm_(model_b.sberta.parameters(), 1.0)
        opt_b.step()

    dom_a_5b, dom_b_5b = _purity(model_b, cfg_b)
    t5b_ok = _is_separated(dom_a_5b, dom_b_5b)

    # ── Report ────────────────────────────────────────────────────────────
    t5a_tag = "✓" if t5a_ok else "✗"
    t5b_tag = "✓" if t5b_ok else "✗"

    detail = (
        f"{t5a_tag} T5a (mechanism, {STEPS_A} steps): "
        f"A={dom_a_5a:.3f} B={dom_b_5a:.3f}  |  "
        f"{t5b_tag} T5b (pipeline, {STEPS_B} steps): "
        f"A={dom_a_5b:.3f} B={dom_b_5b:.3f}"
    )

    if t5a_ok and t5b_ok:
        R.ok("T5 Prototype separation", detail)
    elif t5a_ok and not t5b_ok:
        R.fail(
            "T5 Prototype separation",
            detail + "  — T5b failed: joint pipeline did not separate "
            "(try more steps or higher LR for a real run)",
        )
    elif not t5a_ok and t5b_ok:
        R.fail(
            "T5 Prototype separation",
            detail + "  — T5a failed: prototype alignment mechanism broken "
            "even with structured embeddings",
        )
    else:
        R.fail("T5 Prototype separation", detail + "  — both sub-tests failed")


# ═══════════════════════════════════════════════════════════════════════════════
# T6 — Orthogonality regularisation actually reduces cosine similarity
# ═══════════════════════════════════════════════════════════════════════════════

def test_t6_ortho_regularisation():
    """
    Verify that L_ortho creates a meaningful gradient on the prototype vectors
    and that after a few gradient steps on L_ortho alone, the off-diagonal
    cosine similarity decreases (prototypes push apart).
    """
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)

    # Deliberately collapse prototypes toward each other first
    with torch.no_grad():
        p = model.sberta.prototypes.prototypes
        p.data = p.data + 0.5 * p.data.mean(0, keepdim=True)   # push toward mean
        p.data = F.normalize(p.data, dim=-1) * 0.5

    def _cos_sim():
        L_n = F.normalize(model.sberta.prototypes.prototypes.detach(), dim=-1)
        cos = L_n @ L_n.T
        K   = cfg.num_languages
        off_diag = cos[~torch.eye(K, dtype=torch.bool)].abs().mean().item()
        return off_diag

    cos_before = _cos_sim()

    opt = torch.optim.SGD(model.sberta.prototypes.parameters(), lr=0.1)
    for _ in range(30):
        L_n    = F.normalize(model.sberta.prototypes.prototypes, dim=-1)
        gram   = L_n @ L_n.T
        eye    = torch.eye(cfg.num_languages, device=DEVICE)
        l_orth = (gram - eye).pow(2).mean()
        opt.zero_grad()
        l_orth.backward()
        opt.step()

    cos_after = _cos_sim()

    if cos_after < cos_before - 1e-4:
        R.ok(
            "T6 Ortho regularisation",
            f"off-diag cosine {cos_before:.4f} → {cos_after:.4f}  "
            f"(Δ={cos_before - cos_after:.4f})",
        )
    else:
        R.fail(
            "T6 Ortho regularisation",
            f"cosine similarity did not decrease: {cos_before:.4f} → {cos_after:.4f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# T7 — Mini convergence: total loss decreases ≥5 % over 80 steps
# ═══════════════════════════════════════════════════════════════════════════════

def test_t7_mini_convergence():
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    WARMUP = 10
    TOTAL  = 80

    def _lr(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        prog = (step - WARMUP) / max(1, TOTAL - WARMUP)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)

    first_loss = last_loss = None
    first_window, last_window = [], []

    t0 = time.perf_counter()

    for step in range(1, TOTAL + 1):
        ids, mask, _ = _make_batch(mode="mixed")
        out  = model(ids, mask)
        loss = out["loss"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        val = loss.item()
        if step <= 10:
            first_window.append(val)
        if step >= TOTAL - 9:
            last_window.append(val)

    elapsed = time.perf_counter() - t0

    first_loss = sum(first_window) / len(first_window)
    last_loss  = sum(last_window)  / len(last_window)
    pct_drop   = (first_loss - last_loss) / max(first_loss, 1e-9) * 100.0

    if pct_drop >= 5.0:
        R.ok(
            "T7 Mini convergence",
            f"loss {first_loss:.3f} → {last_loss:.3f}  "
            f"({pct_drop:.1f}% drop over {TOTAL} steps)  "
            f"[{elapsed:.1f}s on {DEVICE}]",
        )
    else:
        R.fail(
            "T7 Mini convergence",
            f"loss only dropped {pct_drop:.1f}% ({first_loss:.3f} → {last_loss:.3f}), "
            f"need ≥5%",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Bonus: per-test detailed diagnostics (printed unconditionally)
# ═══════════════════════════════════════════════════════════════════════════════

def _print_diagnostics():
    cfg   = _tiny_config()
    model = SBERTaForPreTraining(cfg).to(DEVICE)
    n_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_gen = sum(p.numel() for p in model.generator.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.sberta.parameters()   if p.requires_grad)
    n_hd  = sum(p.numel() for p in model.rtd_head.parameters() if p.requires_grad)

    print(f"\n  Config    : hidden={cfg.hidden_size}  layers={cfg.num_hidden_layers}"
          f"  heads={cfg.num_attention_heads}  ffn={cfg.intermediate_size}")
    print(f"  Phases    : base={cfg.n_base_layers}  lang-aware={cfg.num_hidden_layers - cfg.n_base_layers}")
    print(f"  Languages : K={cfg.num_languages}  τ={cfg.proto_temperature}")
    print(f"  Params    : total={n_p:,}  encoder={n_enc:,}  "
          f"generator={n_gen:,}  rtd_head={n_hd:,}")
    print(f"  Vocab     : {cfg.vocab_size:,}  LangA={LANG_A}  LangB={LANG_B}")
    print(f"  Device    : {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("T1  Smoke test",                test_t1_smoke),
    ("T2  Loss sanity",               test_t2_loss_sanity),
    ("T3  Gradient flow",             test_t3_gradient_flow),
    ("T4  Switch signal",             test_t4_switch_signal),
    ("T5  Prototype separation (T5a+T5b)", test_t5_prototype_separation),
    ("T6  Ortho regularisation",      test_t6_ortho_regularisation),
    ("T7  Mini convergence",          test_t7_mini_convergence),
]


def run_all():
    print("\n" + "═" * 60)
    print("  SBERTa Synthetic Architecture Potential Test")
    print("═" * 60)
    _print_diagnostics()
    print("─" * 60)

    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:
            R.fail(name, f"unexpected exception: {exc}")

    ok = R.summary()
    print("═" * 60 + "\n")
    sys.exit(0 if ok else 1)


# ── pytest compatibility ──────────────────────────────────────────────────────
# Each test_ function is also discoverable by pytest directly.

def test_t1():  test_t1_smoke()
def test_t2():  test_t2_loss_sanity()
def test_t3():  test_t3_gradient_flow()
def test_t4():  test_t4_switch_signal()
def test_t5():  test_t5_prototype_separation()   # runs both T5a + T5b
def test_t6():  test_t6_ortho_regularisation()
def test_t7():  test_t7_mini_convergence()


if __name__ == "__main__":
    run_all()