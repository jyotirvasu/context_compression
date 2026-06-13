"""
LLMLingua Pipeline Demo
------------------------
Demonstrates the full LLMLingua compression pipeline on a math word problem.

Uses simulated perplexity scoring (word-frequency inverse) when the GPT-2
model is unavailable due to network issues. When HuggingFace is reachable,
set USE_REAL_MODEL = True to use actual perplexity-based compression.

Usage:
    python -m Phase_1_LLMLingua.demo
    # or
    cd context_compression && python Phase_1_LLMLingua/demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import tiktoken
from collections import Counter

from Phase_1_LLMLingua.stage_a_budget_controller import BudgetController
from Phase_1_LLMLingua.stage_d_token_compress import TokenCompressionResult
from Phase_1_LLMLingua.stage_e_recovery import ResponseRecovery

# Set True if HuggingFace is reachable and you want real GPT-2 perplexity
USE_REAL_MODEL = False


def simulate_compress(text: str, rate: float = 0.3, force_reserve_digit: bool = True) -> str:
    """
    Simulate LLMLingua token-level compression using word-frequency heuristic.

    In the real pipeline, this is done by computing per-token NLL from GPT-2
    and removing tokens below a percentile threshold.
    """
    tokens = text.split()
    if not tokens:
        return text

    freq = Counter(t.lower().strip(".,!?;:()") for t in tokens)
    total = len(tokens)

    scores = []
    for t in tokens:
        key = t.lower().strip(".,!?;:()")
        score = np.log(total / max(freq[key], 1)) + 0.5
        if force_reserve_digit and any(c.isdigit() for c in t):
            score += 50.0
        scores.append(score)

    n_keep = max(1, int(len(tokens) * rate))
    threshold = sorted(scores, reverse=True)[min(n_keep - 1, len(scores) - 1)]
    kept = [t for t, s in zip(tokens, scores) if s >= threshold]
    return " ".join(kept)


def run_demo():
    tokenizer = tiktoken.get_encoding("cl100k_base")

    contexts = [
        "Sam bought a dozen boxes, each with 30 highlighter pens inside, for 10 dollars each box. "
        "He rearranged five of the boxes into packages of six highlighters each and sold them for "
        "3 dollars per package. He sold the rest of the highlighters separately at the rate of "
        "three pens for 2 dollars. How much profit did he make in total, in dollars?",
        "Lets think step by step. Sam bought 12 boxes x 10 dollars = 120 dollars worth of "
        "highlighters. He bought 12 x 30 = 360 highlighters in total. Sam then took 5 boxes "
        "x 30 = 150 highlighters. He sold these as packages of 6, so 150 / 6 = 25 packages. "
        "He sold them for 25 x 3 = 75 dollars.",
        "After selling these 5 boxes, there were 360 - 150 = 210 highlighters remaining. "
        "These form 210 / 3 = 70 groups of three pens. He sold each of these groups for 2 "
        "dollars each, so he made 70 x 2 = 140 dollars from them. In total, then, he earned "
        "140 + 75 = 215 dollars. Since his original cost was 120, he earned 215 - 120 = 95 "
        "dollars in profit. The answer is 95.",
    ]
    instruction = "Answer the following math word problem step by step."
    question = "How much profit did Sam make?"
    rate = 0.3

    print("=" * 60)
    print(f"LLMLINGUA PIPELINE DEMO (rate={rate})")
    print("=" * 60)
    print()

    original_full = "\n\n".join([instruction] + contexts + [question])
    original_tokens = len(tokenizer.encode(original_full))

    # ─── Stage A: Budget Controller ───
    print("[Stage A] Budget Controller")
    bc = BudgetController({"rate": rate, "target_token": -1, "context_budget": "+100"})
    budget = bc.allocate(instruction, contexts, question)
    print(f"  Original tokens:     {original_tokens}")
    print(f"  Target total:        {budget.total_target}")
    print(f"  Instruction budget:  {budget.instruction_tokens} (preserved)")
    print(f"  Question budget:     {budget.question_tokens} (preserved)")
    print(f"  Context budget:      {budget.context_budget}")
    print(f"  Per-context budgets: {budget.per_context_budget}")
    print()

    # ─── Stage B: Context-Level Filter ───
    print("[Stage B] Context-Level Filter")
    if USE_REAL_MODEL:
        from Phase_1_LLMLingua.stage_b_context_filter import ContextFilter

        cf = ContextFilter({"enabled": True, "rank_method": "llmlingua", "model_name": "gpt2", "device": "cpu"})
        ranked, selected = cf.filter(contexts, question)
        selected_contexts = [rc.text for rc in ranked]
        print(f"  Selected {len(selected)}/{len(contexts)} contexts by perplexity")
    else:
        selected_contexts = contexts
        print(f"  [Simulated] All {len(contexts)} contexts retained")
    print()

    # ─── Stage C: Sentence-Level Filter ───
    print("[Stage C] Sentence-Level Filter: DISABLED")
    print()

    # ─── Stage D: Token-Level Compression ───
    print("[Stage D] Token-Level Compression")
    if USE_REAL_MODEL:
        from Phase_1_LLMLingua.stage_d_token_compress import TokenCompressor

        tc = TokenCompressor({"enabled": True, "rate": rate, "model_name": "gpt2", "device": "cpu"})
        compressed_contexts = []
        for i, ctx in enumerate(selected_contexts):
            result = tc.compress(ctx, target_token=budget.per_context_budget[i])
            compressed_contexts.append(result.compressed_text)
            print(f"  Context {i+1}: {result.original_tokens} -> {result.compressed_tokens} tokens ({result.ratio:.1f}x)")
    else:
        compressed_contexts = []
        for i, ctx in enumerate(selected_contexts):
            compressed = simulate_compress(ctx, rate=rate)
            orig_t = len(tokenizer.encode(ctx))
            comp_t = len(tokenizer.encode(compressed))
            print(f"  Context {i+1}: {orig_t} -> {comp_t} tokens ({orig_t/max(comp_t,1):.1f}x)")
            compressed_contexts.append(compressed)
    print()

    # ─── Stage E: Assembly & Statistics ───
    print("[Stage E] Response Recovery & Assembly")
    recovery = ResponseRecovery({"concate_question": True, "add_instruction": True})
    final = recovery.assemble(instruction, compressed_contexts, question, original_tokens)

    print(f"  Original tokens:    {final.origin_tokens}")
    print(f"  Compressed tokens:  {final.compressed_tokens}")
    print(f"  Compression ratio:  {final.ratio}")
    print(f"  Compression rate:   {final.rate}")
    print(f"  Cost saving:        {final.saving}")
    print()
    print("-" * 60)
    print("COMPRESSED PROMPT:")
    print("-" * 60)
    print()
    print(final.compressed_prompt)
    print()

    # ─── Response Recovery Demo ───
    print("-" * 60)
    print("RESPONSE RECOVERY DEMO:")
    print("-" * 60)
    simulated_response = "Sam earned 95 dollars in profit total"
    recovered = recovery.recover(original_full, final.compressed_prompt, simulated_response)
    print(f"  LLM Response:       {simulated_response}")
    print(f"  Recovered Response: {recovered}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
