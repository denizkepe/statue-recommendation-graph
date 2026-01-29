import os
import json
import subprocess
import streamlit as st

st.set_page_config(page_title="Yargıtay Statute Recommender", layout="wide")

st.title("Yargıtay Statute Recommendation Demo")
st.caption("Input: Davacı/vekili dava dilekçesi özeti. Output: BOZMA/ONAMA tahmini + Top-K kanun/madde önerisi.")

DEFAULT_GRAPH = "data/graph_berturk_legal.pt"
DEFAULT_MODEL = "hgt"
DEFAULT_CHAMBER = "9. Hukuk Dairesi"
DEFAULT_TOPK = 10

# ✅ iki checkpoint
DEFAULT_CKPT_OUTCOME = "checkpoints/hgt_berturk-legal_outcome.pt"
DEFAULT_CKPT_STATUTE = "checkpoints/hgt_berturk-legal_statute.pt"

with st.sidebar:
    st.header("Settings")

    graph_path = st.text_input("Graph (.pt)", DEFAULT_GRAPH)
    ckpt_outcome = st.text_input("Checkpoint OUTCOME (.pt)", DEFAULT_CKPT_OUTCOME)
    ckpt_statute = st.text_input("Checkpoint STATUTE (.pt)", DEFAULT_CKPT_STATUTE)

    model_type = st.selectbox("Model", ["gat", "han", "hgt"], index=["gat", "han", "hgt"].index(DEFAULT_MODEL))
    chamber = st.text_input("Chamber", DEFAULT_CHAMBER)
    topk = st.slider("Top-K", 1, 30, DEFAULT_TOPK)

    st.divider()
    st.subheader("Path checks")

    def _check(p, label):
        ok = os.path.exists(p)
        st.write(f"{'✅' if ok else '❌'} {label}: `{p}`")
        return ok

    ok_graph = _check(graph_path, "Graph")
    ok_out = _check(ckpt_outcome, "Outcome ckpt")
    ok_stat = _check(ckpt_statute, "Statute ckpt")

    ready = ok_graph and ok_out and ok_stat

text = st.text_area(
    "Dilekçe/iddia metni",
    value="Davacı vekili dava dilekçesinde ...",
    height=220
)

run = st.button("Predict", type="primary", disabled=not ready)

def run_predict(graph, ckpt_outcome, ckpt_statute, model, chamber, text, topk):
    env = os.environ.copy()
    env["PYTHONPATH"] = "."  # src importları için
    cmd = [
        "python", "src/predict_single.py",
        "--graph", graph,
        "--ckpt-outcome", ckpt_outcome,
        "--ckpt-statute", ckpt_statute,
        "--model", model,
        "--chamber", chamber,
        "--text", text,
        "--topk", str(topk),
        "--json",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.returncode, p.stdout, p.stderr


def outcome_badge(pred: str):
    # küçük “badge” hissi veren HTML
    color = "#22c55e" if pred == "ONAMA" else "#ef4444"
    return f"""
    <div style="
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background:{color}22;
        border:1px solid {color}66;
        color:{color};
        font-weight:700;
        font-size:14px;">
      {pred}
    </div>
    """


if not ready:
    st.warning("Sol menüdeki path check’ler ✅ olana kadar Predict kapalı. (Graph/ckpt yollarını düzelt.)")

if run:
    if not text.strip():
        st.error("Metin boş olamaz.")
    else:
        with st.spinner("Predict çalışıyor..."):
            code, out, err = run_predict(graph_path, ckpt_outcome, ckpt_statute, model_type, chamber, text, topk)

        if code != 0:
            st.error("Predict sırasında hata oluştu.")
            st.code(err or out)
        else:
            try:
                payload = json.loads(out)
            except Exception:
                st.success("Tamamlandı (raw output).")
                st.code(out)
                payload = None

            if payload:
                st.success("Tamamlandı.")

                # ===== OUTCOME UI =====
                pred = payload["outcome"]["pred"]
                p_bozma = float(payload["outcome"]["p_bozma"])
                p_onama = float(payload["outcome"]["p_onama"])

                c1, c2, c3 = st.columns([1.2, 1, 1])
                with c1:
                    st.markdown("### Outcome")
                    st.markdown(outcome_badge(pred), unsafe_allow_html=True)
                with c2:
                    st.metric("p(BOZMA)", f"{p_bozma:.3f}")
                with c3:
                    st.metric("p(ONAMA)", f"{p_onama:.3f}")

                # progress bar: “ONAMA ihtimali”
                st.caption("ONAMA olasılığı (progress):")
                st.progress(min(max(p_onama, 0.0), 1.0))

                st.divider()

                # ===== STATUTES UI =====
                st.markdown("### Statute recommendations")

                recs = payload.get("recommendations", [])
                if not recs:
                    st.info("Öneri bulunamadı.")
                else:
                    # tablo
                    st.dataframe(
                        recs,
                        use_container_width=True,
                        hide_index=True,
                    )

                    # seçilebilir liste (clickable gibi davranır)
                    options = [f"#{r['rank']}  {r['statute_id']}" for r in recs]
                    chosen = st.radio("Detay görmek için bir öneri seç:", options, index=0, horizontal=True)

                    idx = options.index(chosen)
                    r = recs[idx]
                    with st.expander("Seçilen öneri (detay)", expanded=True):
                        st.write("**Statute ID:**", r["statute_id"])
                        st.write("**Law No:**", r["law_no"])
                        st.write("**Article:**", r["article"] if r["article"] else "(yok)")

                st.divider()

                # Debug/trace için JSON’u da koy
                with st.expander("Raw JSON (debug)"):
                    st.code(json.dumps(payload, ensure_ascii=False, indent=2))
