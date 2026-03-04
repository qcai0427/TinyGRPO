"""
Before vs After GRPO Fine-tuning Comparison
Compares Qwen2.5-1.5B-Instruct (base) with our GRPO fine-tuned checkpoint.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FINETUNED_PATH = "./output/step_569"  # Latest saved checkpoint

# ── Test questions: easy and hard ──────────────────────────────────────────────
QUESTIONS = [
    {
        "difficulty": "🟢 简单 (Simple)",
        "q": "What is 7 * 8?",
        "expected": "56",
    },
    {
        "difficulty": "🟡 中等 (Medium)",
        "q": "A train travels at 60 km/h. How far does it travel in 2.5 hours?",
        "expected": "150 km",
    },
    {
        "difficulty": "🔴 较难 (Hard)",
        "q": (
            "Alice has 3 times as many apples as Bob. "
            "If Bob gives Alice 5 apples, Alice will have 5 times as many as Bob. "
            "How many apples does Bob originally have?"
        ),
        "expected": "10",
    },
    {
        "difficulty": "🔴 推理 (Multi-step Reasoning)",
        "q": (
            "A store sells a shirt for $45, which is 25% more than the cost price. "
            "During a sale, the store offers a 10% discount. "
            "What is the store's profit or loss percentage after the discount?"
        ),
        "expected": "12.5% profit",
    },
]

SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


def generate(model, tokenizer, question: str, max_new_tokens=512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,   # greedy for deterministic comparison
            temperature=None,
            top_p=None,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


def run_comparison():
    print(f"\n{'='*70}")
    print("  GRPO Fine-tuning: Before vs After Comparison")
    print(f"  Base model:      {MODEL_NAME}")
    print(f"  Fine-tuned from: {FINETUNED_PATH}")
    print(f"{'='*70}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ──────────── BASE MODEL ────────────
    print("Loading BASE model (no fine-tuning)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )

    base_outputs = []
    for item in QUESTIONS:
        print(f"  [{item['difficulty']}] Generating...")
        out = generate(base_model, tokenizer, item["q"])
        base_outputs.append(out)

    del base_model
    torch.cuda.empty_cache()

    # ──────────── FINE-TUNED MODEL ────────────
    print(f"\nLoading FINE-TUNED model from {FINETUNED_PATH}...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        FINETUNED_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )

    ft_outputs = []
    for item in QUESTIONS:
        print(f"  [{item['difficulty']}] Generating...")
        out = generate(ft_model, tokenizer, item["q"])
        ft_outputs.append(out)

    del ft_model
    torch.cuda.empty_cache()

    # ──────────── PRINT RESULTS ────────────
    print(f"\n\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")

    for i, item in enumerate(QUESTIONS):
        print(f"\n{'─'*70}")
        print(f"  {item['difficulty']} | Expected: {item['expected']}")
        print(f"  Q: {item['q']}")
        print(f"{'─'*70}")
        print(f"\n🔴 BEFORE (Base Qwen2.5-1.5B):\n{base_outputs[i]}")
        print(f"\n🟢 AFTER  (GRPO step_569):\n{ft_outputs[i]}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    run_comparison()
