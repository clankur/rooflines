import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Roofline Model Analysis
    """)
    return


@app.cell
def _(mo):
    model_id_input = mo.ui.text(
        value="mistralai/Mixtral-8x7B-v0.1",
        label="HuggingFace model ID",
        full_width=True,
    )
    fetch_button = mo.ui.run_button(label="Fetch")
    mo.hstack([model_id_input, fetch_button], justify="start", gap=1)
    return fetch_button, model_id_input


@app.cell
def _(fetch_button, model_id_input):
    import urllib.request
    import json as _json

    fetch_button

    def _fetch_hf(model_id):
        try:
            url = f"https://huggingface.co/{model_id}/raw/main/config.json"
            with urllib.request.urlopen(url, timeout=10) as r:
                cfg = _json.loads(r.read())
        except Exception as e:
            return None, str(e)

        total_params = None
        try:
            api_url = f"https://huggingface.co/api/models/{model_id}"
            with urllib.request.urlopen(api_url, timeout=10) as r:
                info = _json.loads(r.read())
            total_params = info.get("safetensors", {}).get("total")
        except Exception:
            pass

        D = cfg.get("hidden_size", 4096)
        F = cfg.get("intermediate_size", D * 4)
        num_layers = cfg.get("num_hidden_layers", 32)
        num_experts = cfg.get("num_local_experts") or cfg.get("n_routed_experts")
        experts_per_tok = cfg.get("num_experts_per_tok")
        n_shared_experts = cfg.get("n_shared_experts", 0)

        if total_params and num_experts and experts_per_tok:
            active_params = total_params * (experts_per_tok + n_shared_experts) / num_experts
        elif total_params:
            active_params = total_params
        else:
            active_params = 2 * num_layers * D * F * (2 if not num_experts else experts_per_tok)

        return {
            "model_id": model_id,
            "hidden_size": D,
            "intermediate_size": F,
            "num_hidden_layers": num_layers,
            "num_experts": num_experts,
            "experts_per_tok": experts_per_tok,
            "total_params": total_params,
            "active_params": active_params,
        }, None

    hf_spec, hf_error = _fetch_hf(model_id_input.value)
    return hf_error, hf_spec


@app.cell
def _(hf_error, hf_spec, mo):
    _out = None
    if hf_error:
        _out = mo.md(f"⚠️ Failed to fetch: `{hf_error}`")
    elif hf_spec:
        moe_info = ""
        if hf_spec["num_experts"]:
            moe_info = f" | MoE: {hf_spec['experts_per_tok']}/{hf_spec['num_experts']} experts"
        params_str = f"{hf_spec['total_params']/1e9:.1f}B" if hf_spec['total_params'] else "unknown"
        _out = mo.md(
            f"**{hf_spec['model_id']}** — "
            f"D={hf_spec['hidden_size']}, F={hf_spec['intermediate_size']}, "
            f"layers={hf_spec['num_hidden_layers']}, "
            f"params={params_str}, "
            f"active≈{hf_spec['active_params']:.2e}"
            f"{moe_info}"
        )
    _out
    return


@app.cell
def _(mo):
    import json as _json

    _REMOTE = "https://raw.githubusercontent.com/clankur/rooflines/main/accelerators.json"

    try:
        from pathlib import Path as _Path
        _specs_path = _Path(__file__).parent / "accelerators.json"
        with open(_specs_path) as f:
            PRESETS = _json.load(f)
    except Exception:
        from pyodide.http import open_url as _open_url
        PRESETS = _json.loads(_open_url(_REMOTE).read())

    chip_dropdown = mo.ui.dropdown(
        options=list(PRESETS.keys()),
        value="H100" if "H100" in PRESETS else list(PRESETS.keys())[0],
        label="Chip preset",
    )
    return PRESETS, chip_dropdown


@app.cell
def _(PRESETS, chip_dropdown, hf_spec, mo):
    def _sci(value, label):
        return mo.ui.text(value=f"{value:.2e}", label=label)

    preset = PRESETS[chip_dropdown.value]

    bf16_flops_input = _sci(preset["bf16_flops"], "Accelerator FLOPs/s")
    hbm_bw_input = _sci(preset["hbm_bw"], "Memory BW (bytes/s)")
    ici_bw_input = _sci(preset["ici_bw"], "Network BW (bytes/s)")

    n_chips_input = mo.ui.slider(
        start=1, stop=16384, value=8, step=1, label="Number of chips (N)", show_value=True
    )
    sharding_dropdown = mo.ui.dropdown(
        options=["Tensor Parallel", "Data Parallel"],
        value="Tensor Parallel",
        label="Sharding strategy",
    )

    D_default = str(hf_spec["hidden_size"]) if hf_spec else "12288"
    F_default = str(hf_spec["intermediate_size"]) if hf_spec else "49152"
    n_active_default = f"{hf_spec['active_params']:.2e}" if hf_spec else "1.00e+12"

    batch_size_input = mo.ui.text(value="256", label="Batch size (B)")
    n_active_input = mo.ui.text(value=n_active_default, label="Active params (N_active)")
    seq_len_input = _sci(4e6, "Sequence length (L)")
    model_dim_input = mo.ui.text(value=D_default, label="Model dim (D)")
    ffn_dim_input = mo.ui.text(value=F_default, label="FFN dim (F)")
    return (
        batch_size_input,
        bf16_flops_input,
        ffn_dim_input,
        hbm_bw_input,
        ici_bw_input,
        model_dim_input,
        n_active_input,
        n_chips_input,
        seq_len_input,
        sharding_dropdown,
    )


@app.cell
def _(
    batch_size_input,
    bf16_flops_input,
    ffn_dim_input,
    hbm_bw_input,
    ici_bw_input,
    model_dim_input,
    n_active_input,
    n_chips_input,
    seq_len_input,
    sharding_dropdown,
):
    def _parse(text_input):
        try:
            return float(text_input.value)
        except ValueError:
            return 1.0

    flops = _parse(bf16_flops_input)
    hbm_bandwidth = _parse(hbm_bw_input)
    ici_bandwidth = _parse(ici_bw_input)
    N = n_chips_input.value
    strategy = sharding_dropdown.value
    B = int(_parse(batch_size_input))
    N_ACTIVE = _parse(n_active_input)
    L = _parse(seq_len_input)
    D = int(_parse(model_dim_input))
    F = int(_parse(ffn_dim_input))
    TOKEN_BYTES_BF16 = 2

    # --- Single-chip roofline ---
    computation_flops = 2 * B * N_ACTIVE
    communication_bytes = N_ACTIVE + B * L * TOKEN_BYTES_BF16

    t_math = computation_flops / flops
    t_memory = communication_bytes / hbm_bandwidth
    t_lower = max(t_math, t_memory)
    t_upper = t_math + t_memory
    bound = "COMPUTE" if t_math > t_memory else "MEMORY"

    ridge_mem = flops / hbm_bandwidth
    ridge_net = flops / ici_bandwidth

    # --- Multi-chip roofline ---
    if strategy == "Tensor Parallel":
        per_chip_flops = 2 * B * D * F / N
        allreduce_bytes = 2 * (N - 1) / N * (2 * B * F) if N > 1 else 0
        net_intensity_val = D / 2 if N > 1 else float('inf')
        net_intensity_label = "D/2"
    else:
        per_chip_flops = 2 * (B / N) * N_ACTIVE
        allreduce_bytes = 2 * (N - 1) / N * (N_ACTIVE * 2) if N > 1 else 0
        net_intensity_val = (B / N) if N > 1 else float('inf')
        net_intensity_label = "B/N"

    t_math_net = per_chip_flops / flops
    t_comms_net = allreduce_bytes / ici_bandwidth if N > 1 else 0
    t_lower_net = max(t_math_net, t_comms_net)
    t_upper_net = t_math_net + t_comms_net
    net_bound = "COMPUTE" if t_math_net > t_comms_net else "NETWORK"

    return (
        B,
        D,
        F,
        L,
        N,
        N_ACTIVE,
        TOKEN_BYTES_BF16,
        allreduce_bytes,
        bound,
        communication_bytes,
        computation_flops,
        flops,
        hbm_bandwidth,
        ici_bandwidth,
        net_bound,
        net_intensity_label,
        net_intensity_val,
        per_chip_flops,
        ridge_mem,
        ridge_net,
        strategy,
        t_comms_net,
        t_lower,
        t_lower_net,
        t_math,
        t_math_net,
        t_memory,
        t_upper,
        t_upper_net,
    )


@app.cell
def _(
    N,
    allreduce_bytes,
    batch_size_input,
    bf16_flops_input,
    bound,
    chip_dropdown,
    communication_bytes,
    computation_flops,
    ffn_dim_input,
    hbm_bw_input,
    ici_bw_input,
    mo,
    model_dim_input,
    n_active_input,
    n_chips_input,
    net_bound,
    net_intensity_label,
    net_intensity_val,
    per_chip_flops,
    ridge_mem,
    ridge_net,
    seq_len_input,
    sharding_dropdown,
    strategy,
    t_comms_net,
    t_lower,
    t_lower_net,
    t_math,
    t_math_net,
    t_memory,
    t_upper,
    t_upper_net,
):
    controls = mo.vstack([
        mo.md("**Chip**"),
        mo.hstack([chip_dropdown, bf16_flops_input, hbm_bw_input, ici_bw_input], justify="start", gap=1),
        mo.md("**Cluster**"),
        mo.hstack([n_chips_input, sharding_dropdown], justify="start", gap=1),
        mo.md("**Workload**"),
        mo.hstack([batch_size_input, n_active_input, seq_len_input, model_dim_input, ffn_dim_input], justify="start", gap=1),
    ])

    single_chip = mo.md(
        f"""
**Single-Chip Roofline**

| | |
|---|---|
| Computation FLOPs | `{computation_flops:.2e}` |
| Communication Bytes | `{communication_bytes:.2e}` |
| $T_{{\\text{{math}}}}$ | `{t_math:.4f} s` |
| $T_{{\\text{{mem}}}}$ | `{t_memory:.4f} s` |
| $T_{{\\text{{exec}}}}$ | `[{t_lower:.4f}, {t_upper:.4f}] s` |
| **Regime** | **{bound}-bound** |
| Memory ridge point | `{ridge_mem:.0f}` FLOPs/byte |
"""
    )

    multi_chip = mo.md(
        f"""
**Multi-Chip Roofline** ({N} chips, {strategy})

| | |
|---|---|
| Per-chip FLOPs | `{per_chip_flops:.2e}` |
| All-reduce bytes | `{allreduce_bytes:.2e}` |
| $T_{{\\text{{math}}}}$ (per chip) | `{t_math_net:.6f} s` |
| $T_{{\\text{{comms}}}}$ | `{t_comms_net:.6f} s` |
| $T_{{\\text{{exec}}}}$ | `[{t_lower_net:.6f}, {t_upper_net:.6f}] s` |
| **Regime** | **{net_bound}-bound** |
| Network ridge point | `{ridge_net:.0f}` FLOPs/byte |
| Intensity ({net_intensity_label}) | `{net_intensity_val:.0f}` |
| Compute-bound when | {net_intensity_label} $> {ridge_net:.0f}$ |
"""
    )

    mo.vstack([
        controls,
        mo.md("---"),
        mo.hstack([single_chip, multi_chip], justify="start", gap=2),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
---

## Background

Sources:

- [How GPT, Claude, and Gemini are actually trained and served – Reiner Pope](https://www.youtube.com/watch?v=xmkSf5IS-zw)
- [JAX Scaling Book — Roofline Analysis](https://jax-ml.github.io/scaling-book/roofline/)

### Where Does the Time Go?

$$
T_{\text{math}} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} \qquad T_{\text{mem}} = \frac{\text{Communication Bytes}}{\text{Memory Bandwidth bytes/s}}
$$

$$
\max(T_{\text{math}}, T_{\text{mem}}) \;\leq\; T_{\text{exec}} \;\leq\; T_{\text{math}} + T_{\text{mem}}
$$

The lower bound assumes perfect overlap; the upper bound assumes zero overlap. Note $T_{\text{upper}} \leq 2 \cdot T_{\text{lower}}$ always.

### Arithmetic Intensity

$$
\text{Intensity(Computation)} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
$$

$$
\text{Intensity(Accelerator)} = \frac{\text{Accelerator FLOPs/s}}{\text{Memory Bandwidth bytes/s}}
$$

We are **compute-bound** when Intensity(Computation) > Intensity(Accelerator).

### Why "2 × N_params" FLOPs per token?

Given input vector of size $K$ and weight matrix $(K, N)$:

$$
\text{einsum}(K,\; KN \to N): \quad K \cdot N \text{ muls} + (K-1) \cdot N \text{ adds} \approx 2KN
$$

For sparse (MoE) models: $\text{FLOPs} \approx 2 \cdot N_{\text{params}} \cdot \frac{\text{experts selected}}{\text{total experts}}$

This drops attention for simplicity — the FFN dominates at short context lengths. In the original lecture Reiner does not include the factor of 2.

### Dot Product Roofline

For $\mathbf{x}, \mathbf{y} \in \text{BF16}^N$: load both vectors ($4N + 2$ bytes), do $2N - 1$ FLOPs.

$$
\text{Intensity(dot product)} \to \frac{1}{2} \quad \text{as } N \to \infty
$$

Always memory-bound — 0.5 FLOPs/byte is far below any modern ridge point.

### Matrix Multiplication Roofline

For $X \in \text{BF16}^{B \times D}$, $Y \in \text{BF16}^{D \times F}$:

$$
\text{Intensity(matmul)} = \frac{BDF}{BD + DF + BF} \approx B \quad (B \ll D, F)
$$

**Rule of thumb:** batch size should exceed the chip's ridge point.

### Network Roofline (Multi-Chip)

**Tensor Parallel** — shard $W \in \mathbb{R}^{D \times F}$ along $D$ across $N$ chips:

- Per-chip compute: $2BDF / N$ FLOPs
- All-reduce (ring): $2 \cdot \frac{N-1}{N} \cdot 2BF$ bytes per chip
- Intensity: $\approx D/2$

Compute-bound when $D/2 > \text{Accelerator FLOPs/s} \;/\; \text{Network Bandwidth bytes/s}$.

**Data Parallel** — replicate model, shard batch across $N$ chips:

- Per-chip compute: $2 \cdot (B/N) \cdot N_{\text{active}}$ FLOPs
- All-reduce gradients (ring): $2 \cdot \frac{N-1}{N} \cdot 2 \cdot N_{\text{active}}$ bytes per chip
- Intensity: $\approx B/N$

Compute-bound when $B/N > \text{Accelerator FLOPs/s} \;/\; \text{Network Bandwidth bytes/s}$.

The ring all-reduce factor $2 \cdot (N-1)/N$ accounts for the reduce-scatter + all-gather phases; it approaches 2 for large $N$.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
