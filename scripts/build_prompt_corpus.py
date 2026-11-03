#!/usr/bin/env python3
"""Build the versioned 200/100 multi-task routing workload (no downloads)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

TASKS = ('qa', 'programming', 'mathematics', 'summarization', 'translation', 'long_context')
LENGTHS = ('short', 'medium', 'long')
SEED = 20261027

# These are authored workloads, not benchmark accuracy questions. Every prompt
# contains its own problem or source text. Test scenarios use disjoint names,
# numerical seeds and document identifiers, fixed before any routing is read.
TRAIN_NAMES = ('Alder', 'Birch', 'Cedar', 'Elm', 'Fir', 'Grove', 'Hazel', 'Iris')
TEST_NAMES = ('Juniper', 'Kestrel', 'Laurel', 'Maple', 'Orchid', 'Pine', 'Quartz', 'Willow')
TOPICS = (
    ('library', 'books', 'reading room', 'librarians', 'catalogue', 'borrowing'),
    ('observatory', 'images', 'telescope', 'astronomers', 'archive', 'observations'),
    ('community garden', 'seedlings', 'greenhouse', 'gardeners', 'seed bank', 'planting'),
    ('bicycle workshop', 'bicycles', 'repair station', 'mechanics', 'parts inventory', 'repairs'),
    ('museum', 'artifacts', 'exhibition hall', 'curators', 'collection register', 'tours'),
    ('water laboratory', 'samples', 'testing bench', 'technicians', 'sample register', 'tests'),
    ('music school', 'instruments', 'practice room', 'teachers', 'booking system', 'lessons'),
    ('weather station', 'readings', 'sensor array', 'observers', 'data archive', 'measurements'),
    ('food cooperative', 'orders', 'packing room', 'volunteers', 'order register', 'deliveries'),
    ('railway archive', 'maps', 'study room', 'archivists', 'index', 'research visits'),
)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def document(name, topic, instance, paragraphs):
    place, items, room, people, register, activity = topic
    rng = random.Random(SEED + instance * 7919)
    budget, initial = rng.randrange(18, 65) * 1000, rng.randrange(15, 55)
    opening = (
        f"Project {name}-{instance} concerns a {place} in the fictional town of {name}. "
        f"The council approved a budget of {budget} credits. The original plan listed "
        f"{initial} {items} and placed the {room} in the east building. "
        f"The project team must maintain the {register} while improving {activity}. "
        "The council distinguishes approved decisions from suggestions and historical notes."
    )
    sections = [f"[S01] {opening}"]
    for p in range(1, paragraphs):
        amount, month = rng.randrange(3, 29), ('March', 'April', 'May', 'June')[p % 4]
        texts = (
            f"During the {month} review, {people} checked {amount} {items}. "
            f"They recorded each finding in the {register}, using a separate entry for damaged items. "
            f"The review counted completed work, rather than projected demand. The coordinator asked "
            f"for a second check before any item could be removed from the public list.",
            f"A visitor survey in {month} received {amount + 32} responses. Some requested longer "
            f"opening hours for the {room}; others preferred clearer instructions. Staff noted that "
            f"these requests had no budget approval. The survey described preferences and did not "
            f"change the adopted schedule for {activity}.",
            f"The training session covered the {register} and the process for handling {items}. "
            f"There were {amount} participants, divided into two small groups. Each group practised "
            f"recording an arrival, a transfer and a correction. The trainers kept the old entries "
            f"so that later reviewers could reconstruct why a record had changed.",
            f"The operations note listed {amount} requests awaiting review. The {people} agreed "
            f"to handle safety issues first, then requests affecting scheduled {activity}. "
            f"Routine cosmetic work would follow. This priority rule applied to the entire project "
            f"and did not authorize additional spending or a move to a different building.",
            f"An equipment inspection found that {amount} labels needed replacement. The team "
            f"assigned a unique identifier to each label and compared it with the {register}. "
            f"They discovered no missing {items}. The inspection report therefore separated "
            f"maintenance costs from purchases and recommended keeping the existing inventory.",
            f"The draft timetable offered {amount} weekly slots for {activity}. It was circulated "
            f"for comment and clearly marked as provisional. The {people} asked for a trial before "
            f"making a permanent commitment. The final schedule would require a signed decision "
            f"from the coordinator and a dated entry in the {register}.",
        )
        sections.append(f"[S{p+1:02d}] {texts[(p + instance) % len(texts)]}")
    # A late authoritative revision creates an actual retrieval/update problem.
    revised = initial + 7 + instance % 13
    sections.insert(max(1, len(sections) * 2 // 3),
                    f"[DECISION-{instance}] The signed final decision supersedes the original plan: "
                    f"use the west building for the {room} and allocate {revised} {items}. "
                    f"The approved budget stays at {budget} credits. Coordinator {name} signed this "
                    f"decision; earlier building and inventory figures are historical.")
    return '\n\n'.join(sections), dict(budget=budget, initial=initial, revised=revised)


def programming_problem(index, split):
    train = (
        "Implement a stable word-frequency counter that ignores case and punctuation.",
        "Implement a bounded least-recently-used cache with get and put operations.",
        "Merge overlapping half-open intervals, explaining how adjacent intervals are treated.",
        "Parse a CSV string with quoted fields and compute totals by product.",
        "Implement breadth-first search to find the shortest path in an unweighted graph.",
        "Write a streaming moving-average generator with a configurable window.",
        "Validate balanced brackets while ignoring bracket characters inside quoted strings.",
        "Implement binary search returning the leftmost matching position.",
        "Group records by date and return the three largest daily totals.",
        "Implement a topological sort and explicitly report a directed cycle.",
        "Write a run-length encoder and decoder that also handle digit characters.",
        "Implement an iterator that merges sorted input iterators without materializing them.",
    )
    test = (
        "Implement a time-to-live cache with an injected clock and lazy expiration.",
        "Find the shortest weighted path using Dijkstra's algorithm and reject negative weights.",
        "Implement a trie supporting insert, exact lookup and prefix lookup.",
        "Build an event deduplicator that retains only the newest timestamp per identifier.",
        "Write a small recursive-descent parser for integers, addition and parentheses.",
        "Compute connected components of an undirected graph including isolated vertices.",
        "Implement a priority queue with stable ordering for equal priorities.",
        "Implement a sliding-window rate limiter with an injected clock.",
        "Rotate a rectangular matrix clockwise and describe the output shape.",
        "Find all anagram groups while preserving the first appearance of each group.",
        "Implement a dependency-aware job readiness tracker with completion notifications.",
        "Write an incremental checksum reader that never loads the whole file in memory.",
    )
    return (train if split == 'train' else test)[index % 12]


def build_prompt(task, length, split, index, instance):
    names = TRAIN_NAMES if split == 'train' else TEST_NAMES
    name = names[index % len(names)]
    topic = TOPICS[(index + (0 if split == 'train' else 3)) % len(TOPICS)]
    paragraphs = dict(short=1, medium=4, long=16)[length]
    doc, facts = document(name, topic, instance, paragraphs)
    if task == 'qa':
        return (f"Read this self-contained record and answer using its evidence.\n\n{doc}\n\n"
                f"Which building is now approved for the {topic[2]}, how many {topic[1]} are "
                "allocated, and did the budget change? Explain which statement takes precedence.\n")
    if task == 'summarization':
        styles = ('five bullet points', 'one concise executive paragraph',
                  'a decision log with decisions, proposals and unresolved work')
        return (f"Summarize the following fictional project record as {styles[index % 3]}. "
                "Preserve numerical facts and distinguish final decisions from provisional proposals.\n\n"
                f"{doc}\n\nSummary:\n")
    if task == 'long_context':
        return (f"Reconcile the statements across this project record.\n\n{doc}\n\n"
                "Answer all parts: (1) Give the original and revised inventory counts and their "
                "difference. (2) Identify the approved location. (3) State the budget. "
                "(4) Explain why a provisional suggestion cannot override the signed decision. "
                "Cite the section labels supporting your answer.\n")
    if task == 'translation':
        if (index // 3) % 3 == 0:
            instruction = "Translate the following English record into Traditional Chinese."
        elif (index // 3) % 3 == 1:
            instruction = "Translate the following English record into Japanese."
        else:
            instruction = "Translate the following English record into French."
        return (f"{instruction} Preserve the section identifiers, names, numbers and distinction "
                f"between proposals and approved decisions. Use natural formal language.\n\n{doc}\n")
    if task == 'programming':
        problem = programming_problem(index, split)
        context = '' if length == 'short' else (
            f"\nApplication background (requirements below take precedence):\n{doc}\n")
        languages = ('Python', 'JavaScript', 'C++17', 'Rust')
        return (f"Programming request {name}-{instance}: use {languages[index % 4]}. {problem} "
                "Include the function or class, representative edge cases, and time/space complexity. "
                "Do not access the network or external libraries.\n" + context +
                f"\nThe implementation must handle empty input and malformed input explicitly. "
                f"Use deterministic ordering. If a capacity or window is needed, use {3 + index % 9}.\n")
    # Arithmetic, algebra, probability, geometry, recurrence and discrete math.
    a, b, c = 12 + instance % 47, 3 + instance % 11, 2 + instance % 7
    problems = (
        f"A vehicle travels {a * b} km in {b * 15} minutes. Find its average speed in km/h.",
        f"Solve the simultaneous equations {b}x + y = {a + b*c} and x - y = {c}.",
        f"A bag has {a} red and {b} blue balls. Draw two without replacement. "
        "Find the probability of exactly one red ball.",
        f"A rectangle has perimeter {2*(a+b)} and its length exceeds its width by {a-b}. "
        "Find its dimensions and area.",
        f"A sequence has u(0)={c} and u(n+1)=2*u(n)+{b}. "
        "Derive a closed form and compute u(5).",
        f"How many distinct arrangements of {b} identical red tiles and {c} identical blue "
        "tiles have no two blue tiles adjacent? State when the answer is zero.",
        f"A store discounts a {a*100}-credit item by {b}% and then adds {c}% tax. "
        "Compute the final price and compare with applying the tax before the discount.",
        f"Minimize x^2 + y^2 subject to x+y={a}. Show the reasoning and the minimum value.",
    )
    # Different numerical instances are held out; problem families are shared.
    context = '' if length == 'short' else (
        f"\nBackground record; only the explicit mathematical problem determines the answer:\n{doc}\n")
    return (f"Mathematics problem {name}-{instance}. {problems[index % len(problems)]} "
            "Show a clear derivation and verify the result.\n" + context + "\nSolution:\n")


def build_records():
    records = []
    for task_index, task in enumerate(TASKS):
        ntrain = 34 if task_index < 2 else 33
        for split, count in (('train', ntrain), ('test', 50 - ntrain)):
            for i in range(count):
                length = LENGTHS[i % 3]
                instance = (task_index + 1) * 10000 + (1000 if split == 'test' else 0) + i
                prompt_id = f'{split}_{task}_{length}_{i:03d}'
                prompt = build_prompt(task, length, split, i, instance)
                records.append(dict(id=prompt_id, split=split, task=task, length_bucket=length,
                                    scenario_id=instance, prompt=prompt, prompt_sha256=digest(prompt),
                                    input_characters=len(prompt), input_words=len(prompt.split())))
    random.Random(SEED).shuffle(records)
    assert len(records) == 300
    assert Counter(r['split'] for r in records) == {'train': 200, 'test': 100}
    assert len({r['prompt_sha256'] for r in records}) == 300
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    records = build_records()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    path = args.output_dir / 'prompts.jsonl'
    if path.exists() and path.read_text() != payload:
        raise ValueError('Refusing to change an existing corpus; use a new version directory')
    path.write_text(payload)
    prompt_dir = args.output_dir / 'prompts'
    prompt_dir.mkdir(exist_ok=True)
    for r in records:
        (prompt_dir / (r['id'] + '.txt')).write_text(r['prompt'])
    manifest = dict(schema_version=1, seed=SEED, corpus_sha256=digest(payload), prompts=300,
                    split_counts=dict(Counter(r['split'] for r in records)),
                    task_counts=dict(Counter(r['task'] for r in records)),
                    length_counts=dict(Counter(r['length_bucket'] for r in records)),
                    cells={f'{s}/{t}/{l}': sum(r['split']==s and r['task']==t and
                           r['length_bucket']==l for r in records)
                           for s in ('train','test') for t in TASKS for l in LENGTHS},
                    provenance='Locally authored, deterministic, self-contained synthetic workload; not an accuracy benchmark.',
                    length_policy='1 / 4 / 16 document paragraphs for short / medium / long; actual tokenizer lengths saved per model.',
                    split_policy='Prompt-level 200/100, stratified by task and length; disjoint scenario IDs, names and numerical instances.',
                    test_use='Never used for expert swaps, g selection, stacking selection or search-budget tuning.',
                    decode_steps=128, prefix_steps=32)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
