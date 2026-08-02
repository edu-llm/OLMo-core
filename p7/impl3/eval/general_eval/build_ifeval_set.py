#!/usr/bin/env python
"""Build the deterministic IFEval prompt set (Zhou et al. 2023 style, rule-verifiable).

Each row = a general-ability prompt with one or more *verifiable* instructions and the
machine-checkable kwargs for ``ifeval_registry``. This mirrors the paper's IFEval: prior-task
instruction-following scored programmatically (no LLM judge / subagents). Every instruction_id
used here has a checker in ``ifeval_registry.REGISTRY``; this script asserts that at build time.

    python build_ifeval_set.py            # -> ifeval_prompts.jsonl
"""
import json
import os

from ifeval_registry import known

OUT = os.path.join(os.path.dirname(__file__), "ifeval_prompts.jsonl")

# (id, prompt, [(instruction_id, kwargs), ...])
SET = [
    ("caps_japan", "Write a short travel tip for visiting Japan. Your entire response must be in all capital letters.",
     [("change_case:english_capital", {})]),
    ("lower_blackhole", "Explain what a black hole is. Your entire response should be in all lowercase letters; use no capital letters.",
     [("change_case:english_lowercase", {})]),
    ("nocomma_season", "Describe your favorite season. Do not use any commas in your response.",
     [("punctuation:no_comma", {})]),
    ("words60_watercycle", "Explain the water cycle. Answer with at least 60 words.",
     [("length_constraints:number_words", {"relation": "at least", "num_words": 60})]),
    ("words40_focus", "Give advice on staying focused while studying. Answer in fewer than 40 words.",
     [("length_constraints:number_words", {"relation": "less than", "num_words": 40})]),
    ("sent3_exercise", "Summarize the benefits of exercise in exactly 3 sentences.",
     [("length_constraints:number_sentences", {"relation": "exactly", "num_sentences": 3})]),
    ("sent4_rainbow", "Explain how rainbows form. Use at least 4 sentences.",
     [("length_constraints:number_sentences", {"relation": "at least", "num_sentences": 4})]),
    ("bullets4_money", "List tips for saving money. Your answer must contain exactly 4 bullet points using markdown, e.g. '* item'.",
     [("detectable_format:number_bullet_lists", {"num_bullets": 4})]),
    ("bullets3_colors", "List the three primary colors as exactly 3 markdown bullet points, e.g. '* red'.",
     [("detectable_format:number_bullet_lists", {"num_bullets": 3})]),
    ("highlight2_sunset", "Describe a sunset. Highlight at least 2 key phrases using markdown asterisks, i.e. *highlighted text*.",
     [("detectable_format:number_highlighted_sections", {"num_highlights": 2})]),
    ("highlight3_backpack", "Write a product description for a backpack. Highlight at least 3 features using markdown asterisks, i.e. *feature*.",
     [("detectable_format:number_highlighted_sections", {"num_highlights": 3})]),
    ("title_dragon", "Write a short story premise about a dragon. Your answer must contain a title wrapped in double angular brackets, e.g. <<Title>>.",
     [("detectable_format:title", {})]),
    ("json_company", "Give the contact details of a fictional company (name, email, phone). Output your entire response as a single valid JSON object and nothing else.",
     [("detectable_format:json_format", {})]),
    ("ps_teacher", "Write a short thank-you note to a teacher. At the end, add a postscript starting with 'P.S.'.",
     [("detectable_content:postscript", {"postscript_marker": "P.S."})]),
    ("pps_dentist", "Write a short reminder about a dentist appointment. At the end, add a postscript that begins with 'P.P.S.'.",
     [("detectable_content:postscript", {"postscript_marker": "P.P.S."})]),
    ("placeholders_email", "Write a template email requesting a meeting. It must contain at least 3 placeholders in square brackets, such as [name].",
     [("detectable_content:number_placeholders", {"num_placeholders": 3})]),
    ("kw_breakfast", "Write two sentences about a healthy breakfast. Include the keywords 'protein' and 'fiber'.",
     [("keywords:existence", {"keywords": ["protein", "fiber"]})]),
    ("kw_gym", "Write a slogan for a gym. Include the words 'strong' and 'today'.",
     [("keywords:existence", {"keywords": ["strong", "today"]})]),
    ("forbid_ocean", "Describe the ocean without using the words 'water' or 'blue'.",
     [("keywords:forbidden_words", {"forbidden_words": ["water", "blue"]})]),
    ("forbid_winter", "Talk about winter without using the word 'cold' or 'snow'.",
     [("keywords:forbidden_words", {"forbidden_words": ["cold", "snow"]})]),
    ("freq_coffee", "Write about your morning routine. The word 'coffee' must appear at least 3 times.",
     [("keywords:frequency", {"keyword": "coffee", "frequency": 3, "relation": "at least"})]),
    ("letter_sea", "Write a sentence about the sea. The letter 's' must appear at least 5 times.",
     [("keywords:letter_frequency", {"letter": "s", "let_frequency": 5, "let_relation": "at least"})]),
    ("end_persist", "Give a motivational message about persistence. Finish your response with the exact phrase: Keep going.",
     [("startend:end_checker", {"end_phrase": "Keep going."})]),
    ("quote_gravity", "Provide a one-sentence definition of gravity. Wrap your entire response in double quotation marks.",
     [("startend:quotation", {})]),
    ("constrained_flat", "Is the earth flat? Answer with exactly one of: 'My answer is yes.', 'My answer is no.', 'My answer is maybe.'",
     [("detectable_format:constrained_response", {})]),
    ("two_catnames", "Give two different names for a pet cat. Give two different responses. Responses and only responses should be separated by six asterisks: ******.",
     [("combination:two_responses", {})]),
    ("sections_photosynthesis", "Explain photosynthesis in 2 sections. Begin each section with the marker 'SECTION' followed by its number.",
     [("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 2})]),
    ("paras2_city", "Describe a city you would like to visit. Your response must contain exactly 2 paragraphs, separated by a blank line.",
     [("length_constraints:number_paragraphs", {"num_paragraphs": 2})]),
    ("para_however", "Write 2 paragraphs about dogs, separated by a blank line. The second paragraph must start with the word 'However'.",
     [("length_constraints:nth_paragraph_first_word",
       {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "However"})]),
    ("repeat_smartphone", "First repeat the exact request below, then answer it. Request: List three uses of a smartphone.",
     [("combination:repeat_prompt", {"prompt_to_repeat": "List three uses of a smartphone."})]),
    ("capwords_nasa", "Write a sentence about NASA and its space missions. Include at least 2 fully capitalized words.",
     [("change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"})]),
    # multi-instruction prompts (IFEval frequently combines constraints)
    ("combo_sleep", "Explain why sleep is important. Answer in at least 50 words and do not use any commas.",
     [("length_constraints:number_words", {"relation": "at least", "num_words": 50}),
      ("punctuation:no_comma", {})]),
    ("combo_study", "Give three study tips as exactly 3 markdown bullet points, and finish with the exact phrase: Good luck.",
     [("detectable_format:number_bullet_lists", {"num_bullets": 3}),
      ("startend:end_checker", {"end_phrase": "Good luck."})]),
    ("combo_robot", "Describe a robot in all capital letters, using at least 30 words.",
     [("change_case:english_capital", {}),
      ("length_constraints:number_words", {"relation": "at least", "num_words": 30})]),
]


def main():
    rows = []
    for pid, prompt, instrs in SET:
        ids = [iid for iid, _ in instrs]
        kwargs = [kw for _, kw in instrs]
        for iid in ids:
            assert known(iid), f"no checker registered for {iid}"
        rows.append({"id": pid, "category": "ifeval", "prompt": prompt,
                     "instruction_ids": ids, "kwargs": kwargs})
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_instr = sum(len(r["instruction_ids"]) for r in rows)
    print(f"wrote {len(rows)} prompts ({n_instr} instructions) -> {OUT}")


if __name__ == "__main__":
    main()
