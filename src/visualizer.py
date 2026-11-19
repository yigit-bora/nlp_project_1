"""
Görselleştirme Modülü.
1. Format Dayanıklılık Eğrisi (Sıcaklık vs JSON Başarı Oranı)
2. Sıcaklık vs Kategori Doğruluğu
3. Sıcaklık vs TPS
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from config import FIGURES_DIR, RESULTS_DIR


def set_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
    })


def _get_mode_label(df: pd.DataFrame) -> str:
    """DataFrame içindeki mode/setting sütunundan okunabilir ortam etiketi üretir."""
    for col in ("mode", "setting"):
        if col in df.columns and len(df) > 0:
            val = str(df[col].iloc[0]).strip().lower()
            if val == "schema_constrained":
                return "Schema-Constrained (format=JSON_SCHEMA)"
            if val == "prompt_only":
                return "Prompt-Only (format kapalı)"
    return ""


def _set_temp_axes(ax, temps):
    ax.set_xticks(temps)
    if temps:
        ax.set_xlim(min(temps) - 0.08, max(temps) + 0.12)


def plot_json_success_curve(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Sıcaklık vs Saf JSON ve Sözdizimi Geçerliliği."""
    set_style()
    save_path = save_path or (FIGURES_DIR / "json_success_vs_temperature.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("temperature")
    temps = df["temperature"].tolist()

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    pure_col = "pure_json_rate" if "pure_json_rate" in df.columns else "json_success_rate"
    ax.plot(temps, df[pure_col],
            marker="o", linewidth=2.5, markersize=8, color="#e74c3c",
            label="Saf JSON Başarısı (Sıfır Tolerans)")

    has_syntax = "syntax_valid_rate" in df.columns
    if has_syntax:
        ax.plot(temps, df["syntax_valid_rate"],
                marker="s", linewidth=2.5, markersize=8, color="#2ecc71", linestyle="--",
                label="Yapısal Sözdizimi Geçerliliği (Temizleme Sonrası)")

    for _, row in df.iterrows():
        t = row["temperature"]
        p_val = row[pure_col]
        if has_syntax:
            s_val = row["syntax_valid_rate"]
            if abs(s_val - p_val) < 5:
                # İkisi aynı/çok yakın ise birini üste birini alta yaz
                ax.annotate(f"%{p_val:.1f}", (t, p_val), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=9.5, fontweight="bold", color="#c0392b")
                ax.annotate(f"%{s_val:.1f}", (t, s_val), textcoords="offset points", xytext=(0, -16),
                            ha="center", fontsize=9.5, fontweight="bold", color="#27ae60")
            else:
                p_off = (0, 9) if p_val >= s_val else (0, -16)
                s_off = (0, -16) if p_val >= s_val else (0, 9)
                ax.annotate(f"%{p_val:.1f}", (t, p_val), textcoords="offset points", xytext=p_off,
                            ha="center", fontsize=9.5, fontweight="bold", color="#c0392b")
                ax.annotate(f"%{s_val:.1f}", (t, s_val), textcoords="offset points", xytext=s_off,
                            ha="center", fontsize=9.5, fontweight="bold", color="#27ae60")
        else:
            ax.annotate(f"%{p_val:.1f}", (t, p_val), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9.5, fontweight="bold", color="#c0392b")

    mode_lbl = _get_mode_label(df)
    subtitle = f"Model: {df['model'].iloc[0]}" + (f" | {mode_lbl}" if mode_lbl else "")
    ax.set_title(f"Format Disiplini ve Yapısal Geçerlilik vs Sıcaklık\n{subtitle}")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("Başarı Oranı (%)")
    ax.set_ylim(-8, 114)
    _set_temp_axes(ax, temps)
    ax.legend(loc="lower right" if df[pure_col].mean() > 50 else "center right", fontsize=9.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 JSON Başarı Eğrisi: {save_path}")
    return save_path


def plot_accuracy_curve(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Sıcaklık vs Kurtarılmış (Parsed) ve Uçtan Uca (Strict) Doğruluk."""
    set_style()
    save_path = save_path or (FIGURES_DIR / "accuracy_vs_temperature.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("temperature")
    temps = df["temperature"].tolist()

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)

    parsed_cat_col = "parsed_category_accuracy" if "parsed_category_accuracy" in df.columns else "category_accuracy"
    parsed_sent_col = "parsed_sentiment_accuracy" if "parsed_sentiment_accuracy" in df.columns else "sentiment_accuracy"
    strict_cat_col = "strict_category_accuracy" if "strict_category_accuracy" in df.columns else None
    strict_sent_col = "strict_sentiment_accuracy" if "strict_sentiment_accuracy" in df.columns else None

    ax.plot(temps, df[parsed_cat_col],
            marker="s", linewidth=2.5, markersize=8, color="#3498db",
            label="Parsed Kategori Doğruluğu")
    ax.plot(temps, df[parsed_sent_col],
            marker="o", linewidth=2.5, markersize=8, color="#e67e22",
            label="Parsed Duygu Doğruluğu")

    if strict_cat_col:
        ax.plot(temps, df[strict_cat_col],
                marker="s", linewidth=1.5, linestyle="--", color="#85c1e9", alpha=0.8,
                label="Strict Kategori (Saf JSON Şartı)")
    if strict_sent_col:
        ax.plot(temps, df[strict_sent_col],
                marker="o", linewidth=1.5, linestyle="--", color="#f8c471", alpha=0.8,
                label="Strict Duygu (Saf JSON Şartı)")

    for _, row in df.iterrows():
        t = row["temperature"]
        p_cat = row[parsed_cat_col]
        p_sent = row[parsed_sent_col]

        # Parsed etiketler: yüksek olan üstte, düşük olan altta (çakışmasız)
        if p_cat >= p_sent:
            cat_off = (0, 9)
            sent_off = (0, -16)
        else:
            sent_off = (0, 9)
            cat_off = (0, -16)

        ax.annotate(f"%{p_cat:.1f}", (t, p_cat), textcoords="offset points", xytext=cat_off,
                    ha="center", color="#2471a3", fontweight="bold", fontsize=9.5)
        ax.annotate(f"%{p_sent:.1f}", (t, p_sent), textcoords="offset points", xytext=sent_off,
                    ha="center", color="#d35400", fontweight="bold", fontsize=9.5)

        # Strict değerleri yalnızca Parsed'dan anlamlı farklıysa (≥ 4 puan) yaz
        if strict_cat_col and strict_sent_col:
            s_cat = row[strict_cat_col]
            s_sent = row[strict_sent_col]
            if abs(p_cat - s_cat) >= 4.0:
                off = (0, -15) if s_cat <= s_sent else (0, 8)
                ax.annotate(f"%{s_cat:.1f}", (t, s_cat), textcoords="offset points", xytext=off,
                            ha="center", color="#5499c7", fontweight="bold", fontsize=8.5)
            if abs(p_sent - s_sent) >= 4.0:
                off = (0, 8) if s_sent > s_cat else (0, -15)
                ax.annotate(f"%{s_sent:.1f}", (t, s_sent), textcoords="offset points", xytext=off,
                            ha="center", color="#e59866", fontweight="bold", fontsize=8.5)

    mode_lbl = _get_mode_label(df)
    subtitle = f"Model: {df['model'].iloc[0]}" + (f" | {mode_lbl}" if mode_lbl else "")
    ax.set_title(f"Sıcaklık Stres Testi vs Sınıflandırma Doğruluğu\n{subtitle}")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("Doğruluk Oranı (%)")
    ax.set_ylim(-8, 108)
    _set_temp_axes(ax, temps)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Doğruluk Eğrisi: {save_path}")
    return save_path


def plot_dashboard(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """3 panelli özet dashboard."""
    set_style()
    save_path = save_path or (FIGURES_DIR / "dashboard.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("temperature").reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=150)
    temps = df["temperature"].tolist()

    # Panel 1: Format Disiplini
    pure_col = "pure_json_rate" if "pure_json_rate" in df.columns else "json_success_rate"
    axes[0].plot(temps, df[pure_col], "o-", linewidth=2, color="#e74c3c", markersize=8, label="Saf JSON")
    has_syntax = "syntax_valid_rate" in df.columns
    if has_syntax:
        axes[0].plot(temps, df["syntax_valid_rate"], "s--", linewidth=2, color="#2ecc71", markersize=8, label="Sözdizimi Geçerli")
    for _, row in df.iterrows():
        t = row["temperature"]
        p_val = row[pure_col]
        if has_syntax:
            s_val = row["syntax_valid_rate"]
            if abs(s_val - p_val) < 6:
                axes[0].annotate(f"%{p_val:.0f}", (t, p_val), textcoords="offset points", xytext=(0, 8),
                                 ha="center", fontsize=8.5, fontweight="bold", color="#c0392b")
                axes[0].annotate(f"%{s_val:.0f}", (t, s_val), textcoords="offset points", xytext=(0, -14),
                                 ha="center", fontsize=8.5, fontweight="bold", color="#27ae60")
            else:
                axes[0].annotate(f"%{p_val:.0f}", (t, p_val), textcoords="offset points", xytext=(0, -14),
                                 ha="center", fontsize=8.5, fontweight="bold", color="#c0392b")
                axes[0].annotate(f"%{s_val:.0f}", (t, s_val), textcoords="offset points", xytext=(0, 8),
                                 ha="center", fontsize=8.5, fontweight="bold", color="#27ae60")
    axes[0].set_title("Format & Sözdizimi Başarısı (%)")
    axes[0].set_xlabel("Sıcaklık")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(-8, 114)
    _set_temp_axes(axes[0], temps)
    axes[0].legend(loc="best", fontsize=8.5)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Sınıflandırma Doğruluğu
    parsed_cat_col = "parsed_category_accuracy" if "parsed_category_accuracy" in df.columns else "overall_category_accuracy"
    parsed_sent_col = "parsed_sentiment_accuracy" if "parsed_sentiment_accuracy" in df.columns else "overall_sentiment_accuracy"
    strict_cat_col = "strict_category_accuracy" if "strict_category_accuracy" in df.columns else None
    strict_sent_col = "strict_sentiment_accuracy" if "strict_sentiment_accuracy" in df.columns else None

    axes[1].plot(temps, df[parsed_cat_col], "s-", linewidth=2.5, color="#3498db", markersize=8, label="Parsed Kategori")
    axes[1].plot(temps, df[parsed_sent_col], "o-", linewidth=2.5, color="#e67e22", markersize=8, label="Parsed Duygu")
    if strict_cat_col:
        axes[1].plot(temps, df[strict_cat_col], "s--", linewidth=1.5, color="#85c1e9", markersize=5, alpha=0.8, label="Strict Kategori")
    if strict_sent_col:
        axes[1].plot(temps, df[strict_sent_col], "o--", linewidth=1.5, color="#f8c471", markersize=5, alpha=0.8, label="Strict Duygu")

    for _, row in df.iterrows():
        t = row["temperature"]
        p_cat = row[parsed_cat_col]
        p_sent = row[parsed_sent_col]
        cat_off = (0, 8) if p_cat >= p_sent else (0, -15)
        sent_off = (0, -15) if p_cat >= p_sent else (0, 8)
        axes[1].annotate(f"%{p_cat:.0f}", (t, p_cat), textcoords="offset points", xytext=cat_off,
                         ha="center", fontsize=8.5, color="#2471a3", fontweight="bold")
        axes[1].annotate(f"%{p_sent:.0f}", (t, p_sent), textcoords="offset points", xytext=sent_off,
                         ha="center", fontsize=8.5, color="#d35400", fontweight="bold")

        if strict_cat_col and strict_sent_col:
            s_cat = row[strict_cat_col]
            s_sent = row[strict_sent_col]
            if abs(p_cat - s_cat) >= 4.0:
                off = (0, -14) if s_cat <= s_sent else (0, 7)
                axes[1].annotate(f"%{s_cat:.0f}", (t, s_cat), textcoords="offset points", xytext=off,
                                 ha="center", fontsize=8, color="#5499c7", fontweight="bold")
            if abs(p_sent - s_sent) >= 4.0:
                off = (0, 7) if s_sent > s_cat else (0, -14)
                axes[1].annotate(f"%{s_sent:.0f}", (t, s_sent), textcoords="offset points", xytext=off,
                                 ha="center", fontsize=8, color="#e59866", fontweight="bold")

    axes[1].set_title("Sınıflandırma Doğruluğu (Parsed & Strict)")
    axes[1].set_xlabel("Sıcaklık")
    axes[1].set_ylabel("Doğruluk (%)")
    axes[1].set_ylim(-8, 108)
    _set_temp_axes(axes[1], temps)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: TPS
    max_tps = max(df["mean_tps"].max(), 10)
    axes[2].bar(range(len(temps)), df["mean_tps"], color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"][:len(temps)])
    axes[2].set_title("Ortalama Üretim Hızı (TPS)")
    axes[2].set_xlabel("Sıcaklık")
    axes[2].set_ylabel("Tokens / Saniye")
    axes[2].set_ylim(0, max_tps * 1.15)
    axes[2].set_xticks(range(len(temps)))
    axes[2].set_xticklabels([str(t) for t in temps])
    for i, r in df.iterrows():
        axes[2].text(i, r["mean_tps"] + max_tps * 0.02, f"{r['mean_tps']:.1f}", ha="center", fontsize=9.5, fontweight="bold")

    mode_lbl = _get_mode_label(df)
    title_suffix = f" | {mode_lbl}" if mode_lbl else ""
    plt.suptitle(f"LLM Stres Testi Dashboard — {df['model'].iloc[0]}{title_suffix}", fontsize=13.5, fontweight="bold")
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Dashboard: {save_path}")
    return save_path


def _plot_entity_prf_curve(
    df: pd.DataFrame,
    prefix: str,
    title_label: str,
    f1_color: str,
    f1_text_color: str,
    p_color: str,
    r_color: str,
    marker: str,
    save_path: Path,
) -> Path:
    """Exact veya Partial Entity F1 + Precision + Recall eğrilerini çakışmasız etiketlerle çizer."""
    set_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("temperature")
    temps = df["temperature"].tolist()

    fig, ax = plt.subplots(figsize=(8.8, 5.4), dpi=150)

    f1_col = f"{prefix}_f1"
    p_col = f"{prefix}_precision"
    r_col = f"{prefix}_recall"

    has_f1 = f1_col in df.columns
    has_pr = p_col in df.columns and r_col in df.columns

    if has_f1:
        ax.plot(temps, df[f1_col], marker=marker, linewidth=2.8, markersize=8,
                color=f1_color, label=f"{title_label} F1")
    if has_pr:
        ax.plot(temps, df[p_col], marker="^", linewidth=1.6, linestyle="--",
                color=p_color, alpha=0.85, label=f"{title_label} Precision (Hassasiyet)")
        ax.plot(temps, df[r_col], marker="v", linewidth=1.6, linestyle="--",
                color=r_color, alpha=0.85, label=f"{title_label} Recall (Duyarlılık)")

    # Üç seriyi dikey değere göre sıralayıp çakışmasız etiketle
    for _, row in df.iterrows():
        t = row["temperature"]
        points = []
        if has_r := (has_pr):
            points.append(("R", float(row[r_col]), r_color, 9))
        if has_f1:
            points.append(("F1", float(row[f1_col]), f1_text_color, 10))
        if has_pr:
            points.append(("P", float(row[p_col]), p_color, 9))

        points.sort(key=lambda x: x[1])  # Küçükten büyüğe
        if len(points) == 3:
            low, mid, high = points
            # En alt: aşağıya; En üst: yukarıya; Orta: boşluğa göre yerleştir
            ax.annotate(f"%{low[1]:.1f}", (t, low[1]), textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=low[3], color=low[2], fontweight="bold")
            ax.annotate(f"%{high[1]:.1f}", (t, high[1]), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=high[3], color=high[2], fontweight="bold")
            gap_up = high[1] - mid[1]
            gap_down = mid[1] - low[1]
            if gap_up >= 7.0:
                mid_off, mid_ha, mid_bbox = (0, 8), "center", None
            elif gap_down >= 7.0:
                mid_off, mid_ha, mid_bbox = (0, -15), "center", None
            else:
                mid_off = (8, 0)
                mid_ha = "left"
                mid_bbox = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)
            ax.annotate(f"%{mid[1]:.1f}", (t, mid[1]), textcoords="offset points", xytext=mid_off,
                        ha=mid_ha, va="center" if mid_ha == "left" else "baseline",
                        fontsize=mid[3], color=mid[2], fontweight="bold", bbox=mid_bbox)
        elif len(points) == 1:
            ax.annotate(f"%{points[0][1]:.1f}", (t, points[0][1]), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=points[0][3], color=points[0][2], fontweight="bold")

    model_name = df["model"].iloc[0] if "model" in df.columns and len(df) > 0 else "LLM"
    mode_lbl = _get_mode_label(df)
    subtitle = f"Model: {model_name}" + (f" | {mode_lbl}" if mode_lbl else "")
    ax.set_title(f"Varlık Çıkarımı: {title_label} vs Sıcaklık\n{subtitle}")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("Skor (%)")
    ax.set_ylim(-6, 105)
    _set_temp_axes(ax, temps)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 {title_label} Grafiği: {save_path}")
    return save_path


def plot_entity_exact_f1_curve(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Sıcaklık vs Exact Match (Birebir Eşleşme) Entity F1, Precision ve Recall Skoru."""
    save_path = save_path or (FIGURES_DIR / "entity_f1_exact_vs_temperature.png")
    return _plot_entity_prf_curve(
        df=df,
        prefix="exact",
        title_label="Exact Match (Birebir Eşleşme)",
        f1_color="#e67e22",
        f1_text_color="#b94a00",
        p_color="#d68910",
        r_color="#9c3d00",
        marker="o",
        save_path=save_path,
    )


def plot_entity_partial_f1_curve(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Sıcaklık vs Partial Match (Kısmi Eşleşme) Entity F1, Precision ve Recall Skoru."""
    save_path = save_path or (FIGURES_DIR / "entity_f1_partial_vs_temperature.png")
    return _plot_entity_prf_curve(
        df=df,
        prefix="partial",
        title_label="Partial Match (Kısmi Eşleşme)",
        f1_color="#27ae60",
        f1_text_color="#145a32",
        p_color="#2ecc71",
        r_color="#117a65",
        marker="s",
        save_path=save_path,
    )


def plot_entity_f1_curve(df: pd.DataFrame, save_path: Optional[Path] = None) -> Path:
    """Sıcaklık vs Exact Match ve Partial Match Entity F1 Skoru (Birleşik)."""
    set_style()
    save_path = save_path or (FIGURES_DIR / "entity_f1_vs_temperature.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    if "exact_f1" in df.columns:
        ax.plot(df["temperature"], df["exact_f1"],
                marker="o", linewidth=2.5, markersize=8, color="#e67e22",
                label="Exact Match F1 (Birebir Eşleşme)")
        for _, row in df.iterrows():
            ax.annotate(f"%{row['exact_f1']:.1f}",
                        (row["temperature"], row["exact_f1"]),
                        textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10, color="#d35400", fontweight="bold")
    
    if "partial_f1" in df.columns:
        ax.plot(df["temperature"], df["partial_f1"],
                marker="s", linewidth=2.5, markersize=8, color="#27ae60",
                label="Partial Match F1 (Kısmi / İçerme Eşleşmesi)")
        for _, row in df.iterrows():
            ax.annotate(f"%{row['partial_f1']:.1f}",
                        (row["temperature"], row["partial_f1"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color="#1e8449", fontweight="bold")

    model_name = df['model'].iloc[0] if "model" in df.columns and len(df) > 0 else "LLM"
    ax.set_title(f"Varlık Çıkarımı F1 Skoru vs Sıcaklık\nModel: {model_name} (Exact Match vs Partial Match)")
    ax.set_xlabel("Sıcaklık (Temperature)")
    ax.set_ylabel("F1 Skoru (%)")
    ax.set_ylim(-5, 105)
    ax.set_xticks(df["temperature"].tolist())
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Entity F1 Eğrisi: {save_path}")
    return save_path


def generate_all_visualizations(
    summary_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> dict:
    csv_path = summary_csv or (RESULTS_DIR / "evaluation_summary.csv")
    df = pd.read_csv(csv_path)

    if output_dir is not None:
        out_dir = Path(output_dir)
    elif "model" in df.columns and len(df) > 0:
        from config import get_model_slug
        slug = get_model_slug(str(df["model"].iloc[0]))
        out_dir = FIGURES_DIR / slug
    else:
        out_dir = FIGURES_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "json_success_vs_temperature.png"
    acc_path = out_dir / "accuracy_vs_temperature.png"
    dash_path = out_dir / "dashboard.png"
    f1_exact_path = out_dir / "entity_f1_exact_vs_temperature.png"
    f1_partial_path = out_dir / "entity_f1_partial_vs_temperature.png"

    results = {
        "json_success": str(plot_json_success_curve(df, save_path=json_path)),
        "accuracy": str(plot_accuracy_curve(df, save_path=acc_path)),
        "dashboard": str(plot_dashboard(df, save_path=dash_path)),
    }

    if "exact_f1" in df.columns or "partial_f1" in df.columns:
        results["entity_f1_exact"] = str(plot_entity_exact_f1_curve(df, save_path=f1_exact_path))
        results["entity_f1_partial"] = str(plot_entity_partial_f1_curve(df, save_path=f1_partial_path))

    return results
