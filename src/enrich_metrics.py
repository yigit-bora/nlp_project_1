"""
Mevcut Sonuçları Zenginleştirici ve Yeni Metrik/Grafik Üretici.
1.200 çıkarımlık mevcut detaylı kayıtlardan (detailed_records.csv):
- Exact Match Entity F1, Precision, Recall
- Partial Match Entity F1, Precision, Recall
- Schema Validity Rate
- Category & Sentiment Macro F1
metriklerini hesaplar, özet tabloları günceller ve Exact vs Partial F1 grafiklerini üretir.
"""

import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import RESULTS_DIR, FIGURES_DIR, DATASET_PATH
from src.cleaner import parse_llm_json
from src.metrics import calculate_entity_f1, calculate_macro_f1
from src.visualizer import (
    generate_all_visualizations,
    plot_entity_f1_curve,
    plot_entity_exact_f1_curve,
    plot_entity_partial_f1_curve,
    set_style
)


def write_mode_tables_md(m_slug: str, mode_name: str, df: pd.DataFrame, out_path: Path) -> None:
    """Her model ve ortam için ilk bakışta okunabilir Markdown özet tabloları oluşturur."""
    model_name = str(df["model"].iloc[0])
    mode_title = "Prompt-Only (Ollama `format` parametresi KAPALI)" if mode_name == "prompt_only" else "Schema-Constrained (Ollama `format = JSON_SCHEMA`)"
    samples = int(df["total_samples"].iloc[0])

    lines = [
        f"# {model_name} — {mode_title}",
        "",
        f"- **Model Slug:** `{m_slug}` | **Quantization:** `Q4_K_M` | **Seed:** `42` | **Num Ctx:** `4096`",
        f"- **Ortam:** `{mode_name}` | **Örneklem:** Sıcaklık başına `{samples}` örnek (Toplam `{samples * len(df)}` çıkarım)",
        f"- **Ham Veri:** [{mode_name}/evaluation_summary.csv](./{mode_name}/evaluation_summary.csv) | [{mode_name}/detailed_records.csv](./{mode_name}/detailed_records.csv)",
        "",
        "## 1. Format Disiplini, Şema Geçerliliği ve Hız",
        "",
        "| Sıcaklık ($T$) | Ham JSON (`pure_json_rate`) | Temizlenmiş JSON (`syntax_valid_rate`) | Şema Geçerliliği (`schema_valid_rate`) | Ort. Entity / Örnek | Ort. TPS |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, r in df.sort_values("temperature").iterrows():
        lines.append(
            f"| {r['temperature']} | %{r['pure_json_rate']:.2f} | %{r['syntax_valid_rate']:.2f} | "
            f"%{r['schema_valid_rate']:.2f} | {r['avg_entities_per_sample']:.2f} | {r['mean_tps']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. Sınıflandırma Doğruluğu (Accuracy) ve Macro F1",
        "",
        "| Sıcaklık ($T$) | Strict Kategori Acc | Parsed Kategori Acc | Kategori Macro F1 | Strict Duygu Acc | Parsed Duygu Acc | Duygu Macro F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df.sort_values("temperature").iterrows():
        lines.append(
            f"| {r['temperature']} | %{r['strict_category_accuracy']:.2f} | %{r['parsed_category_accuracy']:.2f} | "
            f"%{r['category_macro_f1']:.2f} | %{r['strict_sentiment_accuracy']:.2f} | "
            f"%{r['parsed_sentiment_accuracy']:.2f} | %{r['sentiment_macro_f1']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. Varlık Çıkarımı (Entity Extraction) — Exact & Partial Match",
        "",
        "| Sıcaklık ($T$) | Exact Precision | Exact Recall | Exact F1 | Partial Precision | Partial Recall | Partial F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for _, r in df.sort_values("temperature").iterrows():
        lines.append(
            f"| {r['temperature']} | %{r['exact_precision']:.2f} | %{r['exact_recall']:.2f} | **%{r['exact_f1']:.2f}** | "
            f"%{r['partial_precision']:.2f} | %{r['partial_recall']:.2f} | **%{r['partial_f1']:.2f}** |"
        )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def enrich_and_plot():
    set_style()
    
    # 1. Gold Entities yükle
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        gold_map = {item["id"]: item.get("gold_entities", []) for item in json.load(f)}

    models = ["llama3.1_latest", "qwen2.5_1.5b", "llama3.2_latest"]
    all_summaries = []

    for m_slug in models:
        m_dir = RESULTS_DIR / m_slug
        # Ortama özel alt klasörler veya geriye dönük kök klasör
        target_dirs = []
        for mode_name in ["prompt_only", "schema_constrained"]:
            mode_dir = m_dir / mode_name
            if (mode_dir / "detailed_records.csv").exists() and (mode_dir / "evaluation_summary.csv").exists():
                target_dirs.append((mode_name, mode_dir))

        if not any(mn == "prompt_only" for mn, _ in target_dirs):
            if (m_dir / "detailed_records.csv").exists() and (m_dir / "evaluation_summary.csv").exists():
                target_dirs.append(("prompt_only", m_dir))

        if not target_dirs:
            print(f"⚠️ {m_slug} dosyaları bulunamadı, atlanıyor.")
            continue

        for mode_name, target_dir in target_dirs:
            detailed_csv = target_dir / "detailed_records.csv"
            summary_csv = target_dir / "evaluation_summary.csv"

            df_det = pd.read_csv(detailed_csv)
            df_sum = pd.read_csv(summary_csv)

            enriched_rows = []

            for _, s_row in df_sum.iterrows():
                temp = s_row["temperature"]
                group = df_det[df_det["temperature"] == temp]

                preds = []
                golds = []
                schema_valid_count = 0
                cat_preds = []
                cat_gts = []
                sent_preds = []
                sent_gts = []

                for _, r in group.iterrows():
                    sid = r["sample_id"]
                    raw_preview = str(r.get("raw_output_preview", ""))
                    res = parse_llm_json(raw_preview)

                    if res.is_schema_valid:
                        schema_valid_count += 1

                    raw_ents = res.data.get("entities", []) if (res and res.data) else []
                    if not isinstance(raw_ents, list):
                        raw_ents = []
                    preds.append([str(e) for e in raw_ents if e])
                    golds.append(gold_map.get(sid, []))

                    if res.data:
                        cat_preds.append(str(res.data.get("category", "")).strip())
                        cat_gts.append(str(r["gt_category"]).strip())
                        sent_preds.append(str(res.data.get("sentiment", "")).strip())
                        sent_gts.append(str(r["gt_sentiment"]).strip())

                ent_res = calculate_entity_f1(preds, golds)
                cat_f1 = calculate_macro_f1(cat_preds, cat_gts)
                sent_f1 = calculate_macro_f1(sent_preds, sent_gts)
                schema_valid_rate = round((schema_valid_count / len(group)) * 100, 2)

                row_dict = s_row.to_dict()
                row_dict["setting"] = row_dict.get("setting") or mode_name
                row_dict["mode"] = row_dict.get("mode") or mode_name
                row_dict["schema_valid_rate"] = schema_valid_rate
                row_dict["exact_f1"] = ent_res["exact_f1"]
                row_dict["exact_precision"] = ent_res["exact_precision"]
                row_dict["exact_recall"] = ent_res["exact_recall"]
                row_dict["partial_f1"] = ent_res["partial_f1"]
                row_dict["partial_precision"] = ent_res["partial_precision"]
                row_dict["partial_recall"] = ent_res["partial_recall"]
                row_dict["category_macro_f1"] = cat_f1
                row_dict["sentiment_macro_f1"] = sent_f1

                enriched_rows.append(row_dict)

            enriched_df = pd.DataFrame(enriched_rows)
            enriched_df.to_csv(summary_csv, index=False, encoding="utf-8")
            with open(target_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
                json.dump(enriched_rows, f, ensure_ascii=False, indent=2)

            all_summaries.append(enriched_df)

            # İlk bakışta okunabilir Markdown tablo dosyasını model klasörüne yaz
            table_md_path = m_dir / f"{mode_name}_tables.md"
            write_mode_tables_md(m_slug, mode_name, enriched_df, table_md_path)

            # Modele ve ortama özel TÜM grafikleri (5 grafik) güncel stil ve etiketlerle üret
            fig_dir = (FIGURES_DIR / m_slug / mode_name) if target_dir != m_dir else (FIGURES_DIR / m_slug)
            fig_dir.mkdir(parents=True, exist_ok=True)
            generate_all_visualizations(summary_csv=summary_csv, output_dir=fig_dir)

            print(f"✅ [{m_slug} | {mode_name}] Tablolar ({table_md_path.name}) ve 5 grafik güncellendi: {fig_dir}")

    # Master summary tablosunu güncelle
    if all_summaries:
        master_df = pd.concat(all_summaries, ignore_index=True)
        master_path = RESULTS_DIR / "all_models_summary.csv"
        master_df.to_csv(master_path, index=False, encoding="utf-8")
        print(f"📊 Birleşik özet tablosu güncellendi: {master_path}")

        # 3 Modelli Karşılaştırmalı AYRI AYRI Grafikler (ortak_grafikler/ klasörü altına)
        shared_dir = FIGURES_DIR / "ortak_grafikler"
        shared_dir.mkdir(parents=True, exist_ok=True)
        for mode_name in master_df["mode"].unique():
            mode_df = master_df[master_df["mode"] == mode_name]
            suffix = "" if mode_name == "prompt_only" else f"_{mode_name}"
            plot_multi_model_exact_f1(mode_df, shared_dir / f"multi_model_exact_f1{suffix}.png")
            plot_multi_model_partial_f1(mode_df, shared_dir / f"multi_model_partial_f1{suffix}.png")

        if {"prompt_only", "schema_constrained"}.issubset(set(master_df["mode"].unique())):
            plot_prompt_vs_schema_comparison(master_df, shared_dir / "prompt_vs_schema_comparison.png")


def plot_prompt_vs_schema_comparison(df: pd.DataFrame, save_path: Path):
    """Prompt-Only vs Schema-Constrained ortamlarını 3 modelde (T=0.1) karşılaştırır."""
    set_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), dpi=150)

    models = ["qwen2.5:1.5b", "llama3.2:latest", "llama3.1:latest"]
    labels = ["Qwen 2.5\n(1.5B)", "Llama 3.2\n(3B)", "Llama 3.1\n(8B)"]
    x = range(len(models))
    width = 0.36

    df_t = df[df["temperature"] == 0.1]

    metrics = [
        ("pure_json_rate", "Ham (Saf) JSON Başarısı (%) [T=0.1]"),
        ("parsed_category_accuracy", "Kategori Doğruluğu (Parsed, %) [T=0.1]"),
        ("exact_f1", "Entity Exact F1 Skoru (%) [T=0.1]"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        p_vals = []
        s_vals = []
        for m in models:
            p_row = df_t[(df_t["model"] == m) & (df_t["mode"] == "prompt_only")]
            s_row = df_t[(df_t["model"] == m) & (df_t["mode"] == "schema_constrained")]
            p_vals.append(float(p_row[col].iloc[0]) if not p_row.empty else 0.0)
            s_vals.append(float(s_row[col].iloc[0]) if not s_row.empty else 0.0)

        b1 = ax.bar([i - width / 2 for i in x], p_vals, width, label="Prompt-Only", color="#e67e22")
        b2 = ax.bar([i + width / 2 for i in x], s_vals, width, label="Schema-Constrained", color="#27ae60")

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.annotate(f"%{h:.1f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 114)
        ax.grid(True, axis="y", alpha=0.3)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, fontsize=10, frameon=True)
    plt.suptitle("Prompt-Only vs. Schema-Constrained (Ollama `format`) Karşılaştırması", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Prompt vs Schema Karşılaştırma Grafiği: {save_path}")


def _annotate_multi_model_points(ax, df: pd.DataFrame, col: str, colors: dict) -> None:
    """Her sıcaklık adımında 3 modelin etiketlerini dikey sıraya göre çakışmasız yerleştirir."""
    for temp in sorted(df["temperature"].unique()):
        sub = df[df["temperature"] == temp]
        pts = [(str(r["model"]), float(r[col])) for _, r in sub.iterrows() if col in r]
        pts.sort(key=lambda item: item[1])  # Küçükten büyüğe
        if len(pts) == 3:
            low, mid, high = pts
            ax.annotate(f"%{low[1]:.1f}", (temp, low[1]), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=9, fontweight="bold", color=colors.get(low[0], "#333"))
            ax.annotate(f"%{high[1]:.1f}", (temp, high[1]), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, fontweight="bold", color=colors.get(high[0], "#333"))
            mid_off = (0, 8) if (high[1] - mid[1]) >= (mid[1] - low[1]) else (0, -15)
            ax.annotate(f"%{mid[1]:.1f}", (temp, mid[1]), textcoords="offset points", xytext=mid_off,
                        ha="center", fontsize=9, fontweight="bold", color=colors.get(mid[0], "#333"))
        else:
            for m_name, val in pts:
                ax.annotate(f"%{val:.1f}", (temp, val), textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=9, fontweight="bold", color=colors.get(m_name, "#333"))


def _mode_title_suffix(df: pd.DataFrame) -> str:
    if "mode" in df.columns and len(df) > 0:
        m = str(df["mode"].iloc[0]).strip().lower()
        if m == "schema_constrained":
            return " (Schema-Constrained)"
        if m == "prompt_only":
            return " (Prompt-Only)"
    return ""


def plot_multi_model_exact_f1(df: pd.DataFrame, save_path: Path):
    """3 modelin Exact Match F1 skorlarını tek başına gösteren bağımsız grafik."""
    set_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=150)

    models = df["model"].unique()
    colors = {"llama3.1:latest": "#27ae60", "qwen2.5:1.5b": "#3498db", "llama3.2:latest": "#e74c3c"}
    markers = {"llama3.1:latest": "o", "qwen2.5:1.5b": "s", "llama3.2:latest": "^"}

    for m in models:
        m_df = df[df["model"] == m].sort_values("temperature")
        if "exact_f1" in m_df.columns:
            ax.plot(m_df["temperature"], m_df["exact_f1"],
                    marker=markers.get(m, "o"), linewidth=2.5, markersize=8,
                    color=colors.get(m, "#333"), label=m)

    _annotate_multi_model_points(ax, df, "exact_f1", colors)
    temps = sorted(df["temperature"].unique())
    ax.set_title(f"3 Model Karşılaştırması: Exact Match (Birebir Eşleşme) F1 Skoru{_mode_title_suffix(df)}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("Exact F1 Skoru (%)")
    ax.set_ylim(10, 72)
    ax.set_xticks(temps)
    ax.set_xlim(min(temps) - 0.08, max(temps) + 0.10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 3 Modelli Ayrı Exact F1 Grafiği: {save_path}")


def plot_multi_model_partial_f1(df: pd.DataFrame, save_path: Path):
    """3 modelin Partial Match F1 skorlarını tek başına gösteren bağımsız grafik."""
    set_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=150)

    models = df["model"].unique()
    colors = {"llama3.1:latest": "#27ae60", "qwen2.5:1.5b": "#3498db", "llama3.2:latest": "#e74c3c"}
    markers = {"llama3.1:latest": "o", "qwen2.5:1.5b": "s", "llama3.2:latest": "^"}

    for m in models:
        m_df = df[df["model"] == m].sort_values("temperature")
        if "partial_f1" in m_df.columns:
            ax.plot(m_df["temperature"], m_df["partial_f1"],
                    marker=markers.get(m, "s"), linewidth=2.5, markersize=8,
                    color=colors.get(m, "#333"), label=m)

    _annotate_multi_model_points(ax, df, "partial_f1", colors)
    temps = sorted(df["temperature"].unique())
    ax.set_title(f"3 Model Karşılaştırması: Partial Match (Kısmi Eşleşme) F1 Skoru{_mode_title_suffix(df)}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("Partial F1 Skoru (%)")
    ax.set_ylim(25, 82)
    ax.set_xticks(temps)
    ax.set_xlim(min(temps) - 0.08, max(temps) + 0.10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9.5)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 3 Modelli Ayrı Partial F1 Grafiği: {save_path}")


def plot_multi_model_f1(df: pd.DataFrame, save_path: Path):
    """3 modelin Exact ve Partial F1 skorlarını yan yana karşılaştıran görsel."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    models = df["model"].unique()
    colors = {"llama3.1:latest": "#27ae60", "qwen2.5:1.5b": "#3498db", "llama3.2:latest": "#e74c3c"}
    markers = {"llama3.1:latest": "o", "qwen2.5:1.5b": "s", "llama3.2:latest": "^"}

    # Sol Panel: Exact Match F1
    for m in models:
        m_df = df[df["model"] == m].sort_values("temperature")
        if "exact_f1" in m_df.columns:
            axes[0].plot(m_df["temperature"], m_df["exact_f1"],
                         marker=markers.get(m, "o"), linewidth=2.2, markersize=7,
                         color=colors.get(m, "#333"), label=m)
            for _, r in m_df.iterrows():
                axes[0].annotate(f"%{r['exact_f1']:.1f}", (r["temperature"], r["exact_f1"]),
                                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    axes[0].set_title("Exact Match (Birebir Eşleşme) F1 Skoru")
    axes[0].set_xlabel("Sıcaklık (Temperature)")
    axes[0].set_ylabel("F1 Skoru (%)")
    axes[0].set_ylim(10, 80)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    # Sağ Panel: Partial Match F1
    for m in models:
        m_df = df[df["model"] == m].sort_values("temperature")
        if "partial_f1" in m_df.columns:
            axes[1].plot(m_df["temperature"], m_df["partial_f1"],
                         marker=markers.get(m, "s"), linewidth=2.2, markersize=7,
                         color=colors.get(m, "#333"), label=m)
            for _, r in m_df.iterrows():
                axes[1].annotate(f"%{r['partial_f1']:.1f}", (r["temperature"], r["partial_f1"]),
                                 textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    axes[1].set_title("Partial Match (Kısmi / İçerme) F1 Skoru")
    axes[1].set_xlabel("Sıcaklık (Temperature)")
    axes[1].set_ylabel("F1 Skoru (%)")
    axes[1].set_ylim(20, 85)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    plt.suptitle("3 Model Karşılaştırmalı Varlık Çıkarımı (Entity Extraction) F1 Skoru", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 3 Modelli Karşılaştırmalı F1 Grafiği: {save_path}")


if __name__ == "__main__":
    enrich_and_plot()
