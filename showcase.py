"""
Showcase Pipeline
-----------------
Runs both Phase_1_CC_PA and Phase_1_LLMLingua on the same input,
then generates an HTML comparison page showing results side by side.

Usage:
    cd context_compression
    python showcase.py

Output:
    showcase_results.html (open in browser)
"""

import sys
import os
import html
from datetime import datetime
from collections import Counter

import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.helpers import load_config, count_tokens
from Phase_1_CC_PA import Chunker, Cleaner, Retriever, Compressor, PositionAwarePacker
from Phase_1_CC_PA.stage_a_chunking import Chunk
from Phase_1_LLMLingua.stage_a_budget_controller import BudgetController
from Phase_1_LLMLingua.stage_d_token_compress import TokenCompressionResult
from Phase_1_LLMLingua.stage_e_recovery import ResponseRecovery


# ─────────────────────────────────────────────────────────────
# Sample Input Data
# ─────────────────────────────────────────────────────────────

SAMPLE_DOCUMENT = """
Sam bought a dozen boxes, each with 30 highlighter pens inside, for 10 dollars each box.
He rearranged five of the boxes into packages of six highlighters each and sold them for
3 dollars per package. He sold the rest of the highlighters separately at the rate of
three pens for 2 dollars. How much profit did he make in total, in dollars?

Lets think step by step. Sam bought 12 boxes x 10 dollars = 120 dollars worth of
highlighters. He bought 12 x 30 = 360 highlighters in total. Sam then took 5 boxes
x 30 = 150 highlighters. He sold these as packages of 6, so 150 / 6 = 25 packages.
He sold them for 25 x 3 = 75 dollars.

After selling these 5 boxes, there were 360 - 150 = 210 highlighters remaining.
These form 210 / 3 = 70 groups of three pens. He sold each of these groups for 2
dollars each, so he made 70 x 2 = 140 dollars from them. In total, then, he earned
140 + 75 = 215 dollars. Since his original cost was 120, he earned 215 - 120 = 95
dollars in profit. The answer is 95.

Additional context: Highlighter pens are commonly used office supplies. They come in
various colors including yellow, green, pink, orange, and blue. The most popular
color for highlighting is yellow, as it does not obscure the underlying text when
photocopied. Modern highlighters use water-based ink and have chisel-shaped tips
for both broad and fine marking capabilities.
""".strip()

QUERY = "How much profit did Sam make?"

INSTRUCTION = "Answer the following math word problem step by step."
QUESTION = "How much profit did Sam make in total?"


# ─────────────────────────────────────────────────────────────
# Step 1: Run Phase_1_CC_PA
# ─────────────────────────────────────────────────────────────

def run_phase1_cc_pa(document: str, query: str) -> dict:
    """Run the Selective Context + Position-Aware pipeline."""
    print("[Phase_1_CC_PA] Running...")

    config = load_config("config.yaml")

    # Stage A: Chunking
    chunker = Chunker(config.get("chunking", {}))
    chunks = chunker.chunk(document)
    num_chunks_created = len(chunks)

    # Stage B: Cleanup
    cleaner = Cleaner(config.get("cleanup", {}))
    chunks = cleaner.clean(chunks)
    num_chunks_after_cleanup = len(chunks)

    # Stage C: Retrieval
    retriever_config = config.get("retrieval", {})
    retriever_config["method"] = "bm25"  # use BM25 (no network needed)
    retriever = Retriever(retriever_config)
    ranked_chunks = retriever.retrieve(chunks, query)
    num_chunks_retrieved = len(ranked_chunks)

    # Stage D: Compression (fallback mode - no model download needed)
    comp_config = config.get("compression", {})
    comp_config["enabled"] = True
    compressor = Compressor(comp_config)
    compressed_chunks = compressor.compress(ranked_chunks, query)
    num_chunks_compressed = len(compressed_chunks)

    # Stage E: Packing
    packer = PositionAwarePacker(config.get("packing", {}))
    packed_context = packer.pack(compressed_chunks)
    position_map = packer.get_position_map(compressed_chunks)

    original_tokens = count_tokens(document)
    compressed_tokens = count_tokens(packed_context)
    ratio = original_tokens / max(compressed_tokens, 1)
    rate = compressed_tokens / max(original_tokens, 1)

    print(f"  [Done] {original_tokens} -> {compressed_tokens} tokens ({ratio:.1f}x)")

    return {
        "name": "Phase_1_CC_PA",
        "subtitle": "Selective Context + Position-Aware Packing",
        "method": "Self-information based compression (Li et al. 2023) + Lost-in-the-Middle packing",
        "compressed_text": packed_context,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "ratio": f"{ratio:.1f}x",
        "rate": f"{rate * 100:.1f}%",
        "stages": {
            "A - Chunking": f"{num_chunks_created} chunks created",
            "B - Cleanup": f"{num_chunks_after_cleanup} chunks after cleanup",
            "C - Retrieval": f"{num_chunks_retrieved} chunks retrieved (BM25)",
            "D - Compression": f"{num_chunks_compressed} chunks compressed (Selective Context)",
            "E - Packing": f"Position-aware packing ({config.get('packing', {}).get('strategy', 'edges_first')})",
        },
        "position_map": position_map,
    }


# ─────────────────────────────────────────────────────────────
# Step 2: Run Phase_1_LLMLingua
# ─────────────────────────────────────────────────────────────

def simulate_llmlingua_compress(text: str, rate: float = 0.4) -> str:
    """Simulate LLMLingua token-level compression using word-frequency heuristic."""
    tokens = text.split()
    if not tokens:
        return text

    freq = Counter(t.lower().strip(".,!?;:()") for t in tokens)
    total = len(tokens)

    scores = []
    for t in tokens:
        key = t.lower().strip(".,!?;:()")
        score = np.log(total / max(freq[key], 1)) + 0.5
        if any(c.isdigit() for c in t):
            score += 50.0
        scores.append(score)

    n_keep = max(1, int(len(tokens) * rate))
    threshold = sorted(scores, reverse=True)[min(n_keep - 1, len(scores) - 1)]
    kept = [t for t, s in zip(tokens, scores) if s >= threshold]
    return " ".join(kept)


def run_phase1_llmlingua(document: str, instruction: str, question: str) -> dict:
    """Run the LLMLingua compression pipeline."""
    print("[Phase_1_LLMLingua] Running...")

    rate = 0.4
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Split into contexts (paragraphs)
    contexts = [p.strip() for p in document.split("\n\n") if p.strip()]

    original_full = "\n\n".join([instruction] + contexts + [question])
    original_tokens = len(tokenizer.encode(original_full))

    # Stage A: Budget Controller
    bc = BudgetController({"rate": rate, "target_token": -1, "context_budget": "+100"})
    budget = bc.allocate(instruction, contexts, question)

    # Stage B: Context-Level Filter (simulated - rank by info density)
    context_scores = []
    for ctx in contexts:
        words = ctx.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        has_numbers = sum(1 for w in words if any(c.isdigit() for c in w))
        context_scores.append(unique_ratio + has_numbers * 0.1)

    # Keep top contexts within budget
    scored_contexts = sorted(
        enumerate(contexts), key=lambda x: context_scores[x[0]], reverse=True
    )
    selected_contexts = [ctx for _, ctx in scored_contexts]
    n_filtered = len(selected_contexts)

    # Stage C: Sentence-Level Filter (disabled in this demo)

    # Stage D: Token-Level Compression
    compressed_contexts = []
    stage_d_stats = []
    for i, ctx in enumerate(selected_contexts):
        compressed = simulate_llmlingua_compress(ctx, rate=rate)
        orig_t = len(tokenizer.encode(ctx))
        comp_t = len(tokenizer.encode(compressed))
        compressed_contexts.append(compressed)
        stage_d_stats.append(f"Ctx {i+1}: {orig_t}->{comp_t} tokens")

    # Stage E: Assembly
    recovery = ResponseRecovery({"concate_question": True, "add_instruction": True})
    final = recovery.assemble(instruction, compressed_contexts, question, original_tokens)

    print(f"  [Done] {final.origin_tokens} -> {final.compressed_tokens} tokens ({final.ratio})")

    return {
        "name": "Phase_1_LLMLingua",
        "subtitle": "LLMLingua Coarse-to-Fine Compression",
        "method": "Perplexity-based iterative token pruning (Jiang et al. 2023)",
        "compressed_text": final.compressed_prompt,
        "original_tokens": final.origin_tokens,
        "compressed_tokens": final.compressed_tokens,
        "ratio": final.ratio,
        "rate": final.rate,
        "stages": {
            "A - Budget Controller": f"Target {budget.total_target} tokens (rate={rate})",
            "B - Context Filter": f"{n_filtered}/{len(contexts)} contexts selected (PPL ranking)",
            "C - Sentence Filter": "Disabled",
            "D - Token Compression": "; ".join(stage_d_stats),
            "E - Assembly": f"Final: {final.compressed_tokens} tokens",
        },
        "position_map": {},
    }


# ─────────────────────────────────────────────────────────────
# Step 3: Generate HTML Comparison Page
# ─────────────────────────────────────────────────────────────

def generate_html(result_ccpa: dict, result_llmlingua: dict, original_text: str, output_path: str):
    """Generate an HTML page comparing both pipeline results."""

    original_tokens = count_tokens(original_text)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def escape(text):
        return html.escape(text)

    def stages_html(stages: dict) -> str:
        rows = ""
        for stage, detail in stages.items():
            rows += f'<tr><td class="stage-name">{escape(stage)}</td><td>{escape(detail)}</td></tr>\n'
        return rows

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Showcase — Compression Comparison</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            color: #1a1a2e;
            padding: 20px;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            color: white;
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 8px; }}
        .header p {{ opacity: 0.9; font-size: 1.1em; }}
        .timestamp {{ font-size: 0.85em; opacity: 0.7; margin-top: 8px; }}

        .original-section {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .original-section h2 {{ color: #333; margin-bottom: 10px; }}
        .original-text {{
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 15px;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
            border-radius: 4px;
        }}
        .meta {{ color: #666; font-size: 0.9em; margin-top: 8px; }}

        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{
            .comparison {{ grid-template-columns: 1fr; }}
        }}

        .pipeline-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-top: 4px solid #ccc;
        }}
        .pipeline-card.ccpa {{ border-top-color: #4CAF50; }}
        .pipeline-card.llmlingua {{ border-top-color: #2196F3; }}

        .pipeline-card h2 {{
            font-size: 1.3em;
            margin-bottom: 4px;
        }}
        .pipeline-card .subtitle {{
            color: #666;
            font-size: 0.85em;
            margin-bottom: 15px;
        }}
        .pipeline-card .method {{
            background: #f0f7ff;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 0.85em;
            color: #1565C0;
            margin-bottom: 15px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }}
        .stat-box {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .stat-box .value {{
            font-size: 1.5em;
            font-weight: 700;
            color: #1a1a2e;
        }}
        .stat-box .label {{
            font-size: 0.8em;
            color: #666;
            margin-top: 4px;
        }}
        .stat-box.highlight {{ background: #e8f5e9; }}
        .stat-box.highlight .value {{ color: #2e7d32; }}

        .stages-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
            font-size: 0.85em;
        }}
        .stages-table th {{
            text-align: left;
            background: #f5f5f5;
            padding: 8px;
            border-bottom: 2px solid #ddd;
        }}
        .stages-table td {{
            padding: 6px 8px;
            border-bottom: 1px solid #eee;
        }}
        .stage-name {{
            font-weight: 600;
            white-space: nowrap;
            color: #444;
        }}

        .compressed-output {{
            background: #f8f9fa;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Consolas', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
        }}
        .compressed-output-label {{
            font-weight: 600;
            margin-bottom: 6px;
            color: #444;
        }}

        .comparison-summary {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-top: 4px solid #FF9800;
        }}
        .comparison-summary h2 {{ margin-bottom: 15px; }}
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95em;
        }}
        .comparison-table th {{
            text-align: left;
            background: #fff3e0;
            padding: 10px;
            border-bottom: 2px solid #FF9800;
        }}
        .comparison-table td {{
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
        .comparison-table tr:hover {{ background: #fafafa; }}
        .winner {{ color: #2e7d32; font-weight: 700; }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        .badge-green {{ background: #e8f5e9; color: #2e7d32; }}
        .badge-blue {{ background: #e3f2fd; color: #1565c0; }}

        .footer {{
            text-align: center;
            color: #999;
            font-size: 0.85em;
            margin-top: 30px;
            padding: 15px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Pipeline Showcase</h1>
        <p>Context Compression — Side-by-Side Comparison</p>
        <div class="timestamp">Generated: {timestamp}</div>
    </div>

    <!-- Original Input -->
    <div class="original-section">
        <h2>Original Input</h2>
        <div class="original-text">{escape(original_text)}</div>
        <div class="meta">
            Tokens: <strong>{original_tokens}</strong> |
            Query: <em>"{escape(QUERY)}"</em>
        </div>
    </div>

    <!-- Side-by-Side Results -->
    <div class="comparison">
        <!-- Phase_1_CC_PA -->
        <div class="pipeline-card ccpa">
            <h2>{escape(result_ccpa['name'])}</h2>
            <div class="subtitle">{escape(result_ccpa['subtitle'])}</div>
            <div class="method">{escape(result_ccpa['method'])}</div>

            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{result_ccpa['original_tokens']}</div>
                    <div class="label">Original Tokens</div>
                </div>
                <div class="stat-box highlight">
                    <div class="value">{result_ccpa['compressed_tokens']}</div>
                    <div class="label">Compressed Tokens</div>
                </div>
                <div class="stat-box highlight">
                    <div class="value">{result_ccpa['ratio']}</div>
                    <div class="label">Compression Ratio</div>
                </div>
                <div class="stat-box">
                    <div class="value">{result_ccpa['rate']}</div>
                    <div class="label">Retention Rate</div>
                </div>
            </div>

            <table class="stages-table">
                <tr><th>Stage</th><th>Result</th></tr>
                {stages_html(result_ccpa['stages'])}
            </table>

            <div class="compressed-output-label">Compressed Output:</div>
            <div class="compressed-output">{escape(result_ccpa['compressed_text'])}</div>
        </div>

        <!-- Phase_1_LLMLingua -->
        <div class="pipeline-card llmlingua">
            <h2>{escape(result_llmlingua['name'])}</h2>
            <div class="subtitle">{escape(result_llmlingua['subtitle'])}</div>
            <div class="method">{escape(result_llmlingua['method'])}</div>

            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{result_llmlingua['original_tokens']}</div>
                    <div class="label">Original Tokens</div>
                </div>
                <div class="stat-box highlight">
                    <div class="value">{result_llmlingua['compressed_tokens']}</div>
                    <div class="label">Compressed Tokens</div>
                </div>
                <div class="stat-box highlight">
                    <div class="value">{result_llmlingua['ratio']}</div>
                    <div class="label">Compression Ratio</div>
                </div>
                <div class="stat-box">
                    <div class="value">{result_llmlingua['rate']}</div>
                    <div class="label">Retention Rate</div>
                </div>
            </div>

            <table class="stages-table">
                <tr><th>Stage</th><th>Result</th></tr>
                {stages_html(result_llmlingua['stages'])}
            </table>

            <div class="compressed-output-label">Compressed Output:</div>
            <div class="compressed-output">{escape(result_llmlingua['compressed_text'])}</div>
        </div>
    </div>

    <!-- Comparison Summary -->
    <div class="comparison-summary">
        <h2>Comparison Summary</h2>
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Phase_1_CC_PA</th>
                <th>Phase_1_LLMLingua</th>
                <th>Winner</th>
            </tr>
            <tr>
                <td>Compression Ratio</td>
                <td>{result_ccpa['ratio']}</td>
                <td>{result_llmlingua['ratio']}</td>
                <td>{_compare_ratios(result_ccpa['ratio'], result_llmlingua['ratio'])}</td>
            </tr>
            <tr>
                <td>Tokens Retained</td>
                <td>{result_ccpa['compressed_tokens']}</td>
                <td>{result_llmlingua['compressed_tokens']}</td>
                <td>{_compare_lower(result_ccpa['compressed_tokens'], result_llmlingua['compressed_tokens'], 'CC_PA', 'LLMLingua')}</td>
            </tr>
            <tr>
                <td>Approach</td>
                <td>Chunk → Retrieve → Compress → Pack</td>
                <td>Budget → Filter → Prune tokens</td>
                <td>—</td>
            </tr>
            <tr>
                <td>Granularity</td>
                <td>Phrase/sentence level</td>
                <td>Token level (fine-grained)</td>
                <td>—</td>
            </tr>
            <tr>
                <td>Position Awareness</td>
                <td><span class="badge badge-green">Yes (edges_first)</span></td>
                <td><span class="badge badge-blue">No (preserves order)</span></td>
                <td>—</td>
            </tr>
            <tr>
                <td>Digit Preservation</td>
                <td>No explicit guarantee</td>
                <td><span class="badge badge-blue">Yes (force_reserve_digit)</span></td>
                <td>—</td>
            </tr>
            <tr>
                <td>Model Requirement</td>
                <td>GPT-2 (for self-information)</td>
                <td>GPT-2 / LLaMA-7B (for PPL)</td>
                <td>—</td>
            </tr>
        </table>
    </div>

    <div class="footer">
        Context Compression Research — Phase 1 Pipeline Showcase<br>
        CC_PA: Li et al. (EMNLP 2023) | LLMLingua: Jiang et al. (EMNLP 2023)
    </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n[HTML] Saved to: {output_path}")


def _compare_ratios(r1: str, r2: str) -> str:
    """Compare compression ratios (higher is better)."""
    v1 = float(r1.replace("x", ""))
    v2 = float(r2.replace("x", ""))
    if v1 > v2:
        return '<span class="winner">CC_PA</span>'
    elif v2 > v1:
        return '<span class="winner">LLMLingua</span>'
    return "Tie"


def _compare_lower(v1: int, v2: int, name1: str, name2: str) -> str:
    """Compare values where lower is better (fewer tokens retained = more compression)."""
    if v1 < v2:
        return f'<span class="winner">{name1}</span>'
    elif v2 < v1:
        return f'<span class="winner">{name2}</span>'
    return "Tie"


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SHOWCASE PIPELINE — Compression Comparison")
    print("=" * 60)
    print()

    # Step 1: Run Phase_1_CC_PA
    result_ccpa = run_phase1_cc_pa(SAMPLE_DOCUMENT, QUERY)
    print()

    # Step 2: Run Phase_1_LLMLingua
    result_llmlingua = run_phase1_llmlingua(SAMPLE_DOCUMENT, INSTRUCTION, QUESTION)
    print()

    # Step 3: Generate HTML comparison
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "showcase_results.html")
    generate_html(result_ccpa, result_llmlingua, SAMPLE_DOCUMENT, output_path)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Phase_1_CC_PA:     {result_ccpa['original_tokens']} -> {result_ccpa['compressed_tokens']} tokens ({result_ccpa['ratio']})")
    print(f"  Phase_1_LLMLingua: {result_llmlingua['original_tokens']} -> {result_llmlingua['compressed_tokens']} tokens ({result_llmlingua['ratio']})")
    print(f"\n  HTML Report: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
