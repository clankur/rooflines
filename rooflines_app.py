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
        value="google/gemma-4-31B",
        label="HuggingFace model ID",
        full_width=True,
    )
    fetch_button = mo.ui.run_button(label="Fetch")
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
def _(batch_size_input, fetch_button, ffn_dim_input, hf_error, hf_spec, mo, model_dim_input, model_id_input, n_active_input):
    import pandas as pd

    search_bar = mo.hstack([model_id_input, fetch_button], justify="start", gap=1, widths=[4, 1])
    workload = mo.hstack([batch_size_input, n_active_input, model_dim_input, ffn_dim_input], justify="start", gap=1)
    left_col = mo.vstack([search_bar, workload])

    _card = None
    if hf_error:
        _card = mo.md(f"⚠️ `{hf_error}`")
    elif hf_spec:
        params_str = f"{hf_spec['total_params']/1e9:.1f}B" if hf_spec['total_params'] else "—"
        moe_str = f"{hf_spec['experts_per_tok']}/{hf_spec['num_experts']}" if hf_spec["num_experts"] else "—"
        df = pd.DataFrame({
            hf_spec["model_id"]: {
                "Hidden dim (D)": hf_spec["hidden_size"],
                "FFN dim (F)": hf_spec["intermediate_size"],
                "Layers": hf_spec["num_hidden_layers"],
                "Params": params_str,
                "Active": f"{hf_spec['active_params']:.2e}",
                "MoE": moe_str,
            }
        })
        _card = mo.ui.table(df, show_column_summaries=False, selection=None)

    mo.hstack([left_col, _card] if _card else [left_col], justify="start", gap=2)
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
    except (NameError, FileNotFoundError):
        from pyodide.http import open_url as _open_url
        PRESETS = _json.loads(_open_url(_REMOTE).read())

    chip_select = mo.ui.multiselect(
        options=list(PRESETS.keys()),
        value=["H100"] if "H100" in PRESETS else [list(PRESETS.keys())[0]],
        label="Chips",
    )
    return PRESETS, chip_select


@app.cell
def _(hf_spec, mo):
    tp_input = mo.ui.slider(start=1, stop=64, value=8, step=1, label="TP", show_value=True)
    dp_input = mo.ui.slider(start=1, stop=4096, value=1, step=1, label="DP", show_value=True)

    D_default = str(hf_spec["hidden_size"]) if hf_spec else "12288"
    F_default = str(hf_spec["intermediate_size"]) if hf_spec else "49152"
    n_active_default = f"{hf_spec['active_params']:.2e}" if hf_spec else "1.00e+12"

    batch_size_input = mo.ui.text(value="256", label="Batch size (B)")
    n_active_input = mo.ui.text(value=n_active_default, label="Active params (N_active)")
    model_dim_input = mo.ui.text(value=D_default, label="Model dim (D)")
    ffn_dim_input = mo.ui.text(value=F_default, label="FFN dim (F)")
    return (
        batch_size_input,
        ffn_dim_input,
        model_dim_input,
        dp_input,
        n_active_input,
        tp_input,
    )


@app.cell
def _(PRESETS, batch_size_input, chip_select, dp_input, ffn_dim_input, model_dim_input, tp_input):
    def _parse(text_input):
        try:
            return float(text_input.value)
        except ValueError:
            return 1.0

    TP = tp_input.value
    DP = dp_input.value
    N = TP * DP
    B = int(_parse(batch_size_input))
    D = int(_parse(model_dim_input))
    F = int(_parse(ffn_dim_input))

    selected = chip_select.value or []

    chip_results = []
    for chip_name in selected:
        spec = PRESETS[chip_name]
        chip_flops = spec["bf16_flops"]
        chip_hbm = spec["hbm_bw"]
        chip_ici = spec["ici_bw"]

        ridge_mem = chip_flops / chip_hbm
        ridge_net = chip_flops / chip_ici

        per_chip_flops = 2 * (B / DP) * D * F / TP
        weight_bytes = 2 * D * F / TP
        allreduce_bytes = 2 * (TP - 1) / TP * (2 * (B / DP) * F) if TP > 1 else 0

        t_math = per_chip_flops / chip_flops
        t_mem = weight_bytes / chip_hbm
        t_net = allreduce_bytes / chip_ici if TP > 1 else 0

        t_lower = max(t_math, t_mem, t_net)
        t_upper = t_math + t_mem + t_net

        if t_math >= t_mem and t_math >= t_net:
            regime = "COMPUTE"
        elif t_net > t_mem:
            regime = "NETWORK"
        else:
            regime = "MEMORY"

        chip_results.append({
            "chip": chip_name,
            "chip_flops": chip_flops,
            "chip_hbm": chip_hbm,
            "chip_ici": chip_ici,
            "t_math": t_math,
            "t_mem": t_mem,
            "t_net": t_net,
            "t_lower": t_lower,
            "t_upper": t_upper,
            "regime": regime,
            "ridge_mem": ridge_mem,
            "ridge_net": ridge_net,
        })

    return chip_results, N


@app.cell
def _(chip_select, dp_input, mo, tp_input):
    mo.vstack([
        chip_select,
        mo.md("**Mesh**"),
        mo.hstack([tp_input, dp_input], justify="start", gap=1),
    ])
    return


@app.cell
def _(chip_results, mo, N):
    import pandas as _pd

    rows = []
    for r in chip_results:
        rows.append({
            "Chip": r["chip"],
            "Chips": N,
            "FLOPs/s": f"{r['chip_flops']:.2e}",
            "HBM BW": f"{r['chip_hbm']:.2e}",
            "ICI BW": f"{r['chip_ici']:.2e}",
            "T_math": f"{r['t_math']:.2e} s",
            "T_mem": f"{r['t_mem']:.2e} s",
            "T_net": f"{r['t_net']:.2e} s" if r["t_net"] > 0 else "—",
            "T_exec": f"[{r['t_lower']:.2e}, {r['t_upper']:.2e}] s",
            "Regime": f"{r['regime']}-bound",
            "Mem Ridge": f"{r['ridge_mem']:.0f}",
            "Net Ridge": f"{r['ridge_net']:.0f}",
        })

    mo.ui.table(_pd.DataFrame(rows), show_column_summaries=False, selection=None) if rows else None
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
