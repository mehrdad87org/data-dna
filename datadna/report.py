from datetime import datetime


def generate_html_report(comparison, output_path: str):
    c = comparison
    bg = "#0D1117"
    card = "#161B22"
    text = "#E6EDF3"
    muted = "#8B949E"
    primary = "#6C5CE7"
    green = "#00D26A"
    red = "#FF4757"
    yellow = "#FFD93D"

    grade_color = green if c.grade.startswith("A") else yellow if c.grade.startswith("B") else red

    dimension_bars = ""
    for dim, score in sorted(c.dimension_scores.items(), key=lambda x: -x[1]):
        pct = score * 100
        bar_color = green if pct > 70 else yellow if pct > 50 else red
        dimension_bars += f"""
        <div style="margin:8px 0;">
            <div style="display:flex; justify-content:space-between; color:{text}; font-size:13px;">
                <span>{dim.replace('_', ' ').title()}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background:#21262D; border-radius:4px; height:8px; margin-top:4px;">
                <div style="background:{bar_color}; height:100%; width:{pct}%; border-radius:4px;"></div>
            </div>
        </div>
        """

    column_scores = ""
    for col, score in sorted(c.per_column_scores.items(), key=lambda x: -x[1])[:15]:
        pct = score * 100
        bar_color = green if pct > 70 else yellow if pct > 50 else red
        column_scores += f"""
        <div style="margin:6px 0;">
            <div style="display:flex; justify-content:space-between; color:{text}; font-size:12px;">
                <span>{col}</span>
                <span>{pct:.1f}%</span>
            </div>
            <div style="background:#21262D; border-radius:3px; height:6px; margin-top:3px;">
                <div style="background:{bar_color}; height:100%; width:{pct}%; border-radius:3px;"></div>
            </div>
        </div>
        """

    mc_section = ""
    if c.mode_collapse:
        collapsed = c.mode_collapse_details.get("collapsed_columns", [])
        mc_section = f"""
        <div style="background:#FF475722; border:1px solid #FF475766; border-radius:8px; padding:16px; margin:16px 0;">
            <h3 style="color:{red}; margin-top:0;">⚠ Mode Collapse Detected</h3>
            <p style="color:{muted};">The following columns show severe mode collapse (coverage < 30%):</p>
            <ul style="color:{text};">{''.join(f'<li>{col}</li>' for col in collapsed)}</ul>
        </div>
        """
    else:
        mc_section = f"""
        <div style="background:#00D26A22; border:1px solid #00D26A66; border-radius:8px; padding:16px; margin:16px 0;">
            <h3 style="color:{green}; margin-top:0;">✓ No Mode Collapse</h3>
            <p style="color:{muted};">All columns maintain adequate unique value coverage.</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataDNA Report</title>
    <style>
        body {{ background:{bg}; color:{text}; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif; margin:0; padding:20px; }}
        .container {{ max-width:1000px; margin:0 auto; }}
        .card {{ background:{card}; border-radius:12px; padding:24px; margin:16px 0; border:1px solid #30363D; }}
    </style>
</head>
<body>
<div class="container">
    <h1 style="text-align:center; color:{text};">DataDNA Analysis Report</h1>

    <div class="card" style="text-align:center;">
        <div style="font-size:72px; font-weight:bold; color:{grade_color};">{c.grade}</div>
        <div style="font-size:24px; color:{text};">Quality Score: {c.quality_score}/100</div>
        <div style="color:{muted};">{c.real_shape[0]} rows × {c.real_shape[1]} cols (real) vs {c.synthetic_shape[0]} rows × {c.synthetic_shape[1]} cols (synthetic)</div>
    </div>

    <div class="card">
        <h2 style="color:{text}; margin-top:0;">DNA Fingerprints</h2>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
            <div>
                <div style="color:{muted}; font-size:12px;">Real Data</div>
                <div style="font-family:monospace; font-size:16px; color:{primary}; word-break:break-all;">{c.real_dna}</div>
            </div>
            <div>
                <div style="color:{muted}; font-size:12px;">Synthetic Data</div>
                <div style="font-family:monospace; font-size:16px; color:{primary}; word-break:break-all;">{c.synthetic_dna}</div>
            </div>
        </div>
        <div style="text-align:center; margin-top:12px; color:{text};">
            DNA Similarity: <strong>{c.dna_similarity:.1%}</strong>
        </div>
    </div>

    <div class="card">
        <h2 style="color:{text}; margin-top:0;">Dimension Scores</h2>
        {dimension_bars}
    </div>

    <div class="card">
        <h2 style="color:{text}; margin-top:0;">ML Utility Score: {c.ml_utility_score}/100</h2>
        <p style="color:{muted};">A model trained on synthetic data achieves {c.ml_utility_score}% of the accuracy of a model trained on real data when tested on real data.</p>
    </div>

    {mc_section}

    <div class="card">
        <h2 style="color:{text}; margin-top:0;">Per-Column Distribution Fidelity</h2>
        {column_scores}
    </div>

    <div style="text-align:center; color:{muted}; margin-top:40px; padding:20px; font-size:13px;">
        Generated by DataDNA v1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
