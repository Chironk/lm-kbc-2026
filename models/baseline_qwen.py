import json
import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForImageTextToText, AutoTokenizer

from models.abstract_model import AbstractModel

NULL_ANSWERS = {"", "none", "n/a", "na", "unknown", "no answer", "empty", "null"}

# Name suffixes that legitimately follow a comma (e.g. "Barack Obama, Jr.")
# -- a comma-split would otherwise turn one correct name into two wrong
# fragments. (The dataset's alias sets always include a suffix-free variant
# too, so this isn't recall-critical, but it removes a spurious extra
# prediction that dings precision every time it happens.)
NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v", "junior", "senior"}

STANDARD_INSTRUCTION = (
    "Answer the question with only the answer(s), as a comma-separated list if there "
    "are multiple, or \"None\" if there is no answer. Do not explain your answer."
)
NEVER_EMPTY_INSTRUCTION = (
    "Answer the question with only the answer(s), as a comma-separated list if there "
    "are multiple. This question always has a real answer for the given subject, so "
    "give your best answer or estimate -- never answer \"None\" or \"unknown\". "
    "Do not explain your answer."
)
FORCE_ANSWER_SUFFIX = (
    "You must not refuse or say \"None\"/\"unknown\" -- give your single best guess."
)
MANY_WINNERS_HINT = (
    "This award has had many winners across its history; list as many correct "
    "winners as you can recall."
)
REASONING_SUFFIX = (
    " First state your answer, then on a new line write \"Reason:\" followed by "
    "one brief, specific sentence justifying it -- a real fact tying the answer "
    "to this exact subject, not a generic statement. If you are only guessing, "
    "say so in the reason."
)
JUDGE_SINGLE_INSTRUCTION_TEMPLATE = (
    "You are shown {n} independent candidate answers (with reasoning) to the same "
    "question, from other copies of yourself. Decide the single best final "
    "answer. Prefer whichever answer is best justified and most consistent "
    "across candidates -- a specific, checkable reason is worth more than "
    "several candidates repeating a generic guess. If the candidates disagree "
    "substantially or their reasoning is weak or generic, answer \"None\" "
    "rather than guessing. Respond with only the final answer (or \"None\") "
    "-- no explanation."
)
JUDGE_LIST_INSTRUCTION_TEMPLATE = (
    "You are shown {n} independent candidate lists (with reasoning) of additional "
    "winners for the same award, from other copies of yourself. Decide which "
    "names are genuinely well-supported -- keep a name only if it has a "
    "specific, checkable justification, or is corroborated across multiple "
    "candidates. Discard names backed only by generic or vague reasoning. "
    "Respond with only the final comma-separated list of confident names (or "
    "\"None\" if none are confident) -- no explanation."
)


def load_prompt_templates(path: str) -> Dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["Relation"], df["PromptTemplate"]))


def load_golden_few_shot_examples(path: str, few_shot: int) -> Dict[str, List[Dict]]:
    """Load hand-curated few-shot exemplars per relation (see
    prompt_templates/golden_few_shot_examples.json). These are deliberately
    picked -- not randomly sampled -- to guarantee: distinct/well-spread
    values for numeric relations (a random draw once gave hasCapacity two
    identical "10000" answers out of 5, anchoring the model into predicting
    "10000" for 56% of val subjects regardless of the actual stadium), and
    coverage of edge cases (empty answers, single vs. multi-object answers)
    for string relations."""
    with open(path) as f:
        golden = json.load(f)
    return {relation: examples[:few_shot] for relation, examples in golden.items()}


def format_answer(object_entities: List[List[str]]) -> str:
    if not object_entities:
        return "None"
    return ", ".join(aliases[0] for aliases in object_entities if aliases)


def parse_answer(text: str, is_numeric: bool) -> List[str]:
    text = text.strip().split("\n")[0].strip()
    text = text.strip(" .")
    if not text or text.lower() in NULL_ANSWERS:
        return []

    if is_numeric:
        match = re.search(r"-?\d[\d,]*\.?\d*", text)
        return [match.group(0)] if match else []

    # Split only on comma/semicolon -- NOT on " and ", since many real
    # entity names contain "and" (e.g. "Bosnia and Herzegovina", a gold
    # countryLandBordersCountry answer; "Bruce Springsteen and The
    # Sessions Band", a gold awardWonBy recipient). Splitting on " and "
    # would silently mangle a correct answer into two wrong fragments.
    # Do strip a leading "and " left over from Oxford-comma lists like
    # "X, Y, and Z" -> ["X", "Y", "and Z"].
    parts = re.split(r",|;", text)
    cleaned = []
    for p in parts:
        p = p.strip(" .")
        if p[:4].lower() == "and ":
            p = p[4:].strip()
        if not p:
            continue
        if cleaned and p.rstrip(".").lower() in NAME_SUFFIXES:
            cleaned[-1] = f"{cleaned[-1]}, {p}"
        else:
            cleaned.append(p)
    return cleaned


class BaselineQwenModel(AbstractModel):
    NUMERIC_RELATIONS = {"hasCapacity", "hasArea"}

    # These relations have zero empty ground-truth answers anywhere in the
    # dataset (every stadium has a capacity, every landmass has an area,
    # every award has been won by someone) -- so letting the model say
    # "None" here is always wrong, never a legitimate abstention.
    NEVER_EMPTY_RELATIONS = {"hasArea", "hasCapacity", "awardWonBy"}

    # Relations whose answer sets are too large for a single bounded
    # generation (e.g. some awards have 200+ recipients) get multi-round
    # "list more" elicitation instead of one-shot generation.
    ITERATIVE_RELATIONS = {"awardWonBy"}

    # Routing for the two self-consistency strategies (see
    # models/baseline_qwen.py commit history / conversation for why these
    # are split): numeric relations have no "why" to reason about -- a
    # generic guess and a well-grounded guess look identical, so a judge
    # adds cost with nothing to judge. Plain N-sample + median is enough.
    # awardWonBy/personHasCityOfDeath fail via confident fabrication, where
    # a specific, checkable reason genuinely distinguishes a real memory
    # from a plausible-sounding guess -- that's exactly what a judge can use
    # that a bare vote count can't.
    NUMERIC_SELF_CONSISTENCY_RELATIONS = {"hasCapacity", "hasArea"}
    SINGLE_SHOT_JUDGE_RELATIONS = {"personHasCityOfDeath"}

    def __init__(self, config: Dict):
        self.config = config
        self.max_new_tokens = config.get("max_new_tokens", 64)
        self.batch_size = config.get("batch_size", 2)
        self.enable_thinking = config.get("enable_thinking", False)
        self.few_shot = config.get("few_shot", 5)
        # Audit P1-1: all 28 golden examples are train subjects, so held-out
        # train targets used to see their own gold answer in-prompt (20/400
        # rows in the S2 train experiment). Enable for any train-side run.
        self.exclude_target_from_shots = config.get(
            "exclude_target_from_shots", False)
        self.reasoning_demo_policy = config.get(
            "reasoning_demo_policy", "require-curated-reason")
        if self.reasoning_demo_policy not in {
                "require-curated-reason", "answer-only"}:
            raise ValueError(
                "reasoning_demo_policy must be 'require-curated-reason' or "
                f"'answer-only', got {self.reasoning_demo_policy!r}")
        self.seed = int(config.get("seed", 42))
        self.model_revision = config.get("model_revision")
        self.raw_records: List[Dict] = []

        # Iterative-elicitation knobs, only used for ITERATIVE_RELATIONS.
        self.award_max_new_tokens = config.get("award_max_new_tokens", 256)
        self.award_max_rounds = config.get("award_max_rounds", 4)

        # Self-consistency knobs for NUMERIC_SELF_CONSISTENCY_RELATIONS:
        # plain N-sample + median, no judge.
        self.numeric_samples = config.get("numeric_samples", 5)
        self.numeric_sample_temperature = config.get("numeric_sample_temperature", 0.8)

        # Candidate + judge knobs for SINGLE_SHOT_JUDGE_RELATIONS and the
        # per-round generation inside ITERATIVE_RELATIONS.
        self.judge_n_samples = config.get("judge_n_samples", 5)
        self.judge_sample_temperature = config.get("judge_sample_temperature", 0.8)
        self.judge_candidate_max_new_tokens = config.get("judge_candidate_max_new_tokens", 96)
        self.judge_max_new_tokens = config.get("judge_max_new_tokens", 64)
        self.judge_list_max_new_tokens = config.get("judge_list_max_new_tokens", 256)
        # awardWonBy candidates carry reasoning *and* a name list in the same
        # generation, competing for tokens -- give that step more room than
        # the judge's own synthesis, which only outputs the filtered subset.
        self.judge_list_candidate_max_new_tokens = config.get(
            "judge_list_candidate_max_new_tokens",
            config.get("award_max_new_tokens", 384),
        )

        self.prompt_templates = load_prompt_templates(config["prompt_templates_file"])
        self.few_shot_examples = load_golden_few_shot_examples(
            config["golden_few_shot_file"], self.few_shot
        )
        if self.reasoning_demo_policy == "require-curated-reason":
            for relation in self.SINGLE_SHOT_JUDGE_RELATIONS | self.ITERATIVE_RELATIONS:
                eligible = sum(bool(example.get("Reason"))
                               for example in self.few_shot_examples.get(relation, []))
                if self.few_shot and eligible == 0:
                    logger.warning(
                        f"{relation}: reasoning_demo_policy=require-curated-reason "
                        "but no selected golden example has a Reason field; "
                        "candidate generation is intentionally zero-shot. Use "
                        "reasoning_demo_policy=answer-only only as a controlled ablation.")

        logger.info(f"Loading model {config['llm_path']} ...")
        quantization_kwargs = {}
        quantization = config.get("quantization")
        if quantization is None and config.get("use_quantization", False):
            quantization = "8bit"  # backward-compat with the old boolean flag
        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif quantization == "4bit":
            from transformers import BitsAndBytesConfig
            quantization_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization not in (None, False):
            raise ValueError(f"Unknown quantization mode: {quantization!r} (expected '4bit', '8bit', or unset)")

        revision_kwargs = ({"revision": self.model_revision}
                           if self.model_revision else {})
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["llm_path"], **revision_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForImageTextToText.from_pretrained(
            config["llm_path"],
            **revision_kwargs,
            dtype=torch.float16,
            device_map="auto",
            max_memory=self._build_max_memory(),
            **quantization_kwargs,
        )
        self.model.eval()

    def _set_subject_seed(self, subject: str, relation: str, phase: str = "") -> int:
        """Stable stochastic seed independent of worker/chunk assignment."""
        payload = f"{self.seed}\0{relation}\0{subject}\0{phase}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
        return value

    def _shot_subjects(self, relation: str, subject: str,
                       with_reasoning: Optional[bool] = None) -> List[str]:
        if with_reasoning is None:
            with_reasoning = (relation in self.SINGLE_SHOT_JUDGE_RELATIONS
                              or relation in self.ITERATIVE_RELATIONS)
        examples = self.few_shot_examples.get(relation, [])
        if (with_reasoning
                and self.reasoning_demo_policy == "require-curated-reason"):
            examples = [example for example in examples if example.get("Reason")]
        shots = [ex["SubjectEntity"] for ex in examples]
        if self.exclude_target_from_shots:
            shots = [s for s in shots if s != subject]
            assert subject not in shots, "target leaked into its own shots"
        return shots

    def _record_raw(self, subject: str, relation: str, prediction: List[str],
                    **payload) -> None:
        payload.setdefault("shot_subjects",
                           self._shot_subjects(relation, subject))
        self.raw_records.append({
            "SubjectEntity": subject,
            "Relation": relation,
            "ObjectEntities": prediction,
            **payload,
        })

    @staticmethod
    def _build_max_memory(reserve_fraction: float = 0.9) -> Optional[Dict]:
        """Cap accelerate's auto device_map at a fraction of each visible
        GPU's capacity (leaving headroom for activations/KV-cache), with
        CPU as an overflow target. Without this, the "auto" balancer's
        static-weight estimate can land a few hundred MB over a device's
        real capacity when splitting across few GPUs -- that shows up as an
        OOM crash rather than a graceful (slower) partial CPU offload."""
        if not torch.cuda.is_available():
            return None
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total_bytes = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_bytes * reserve_fraction)
        max_memory["cpu"] = "64GiB"
        return max_memory

    def _build_prompt(
        self,
        subject_entity: str,
        relation: str,
        force_answer: bool = False,
        already_mentioned: Optional[List[str]] = None,
        with_reasoning: bool = False,
    ) -> str:
        template = self.prompt_templates[relation]
        question = template.format(subject_entity=subject_entity)
        never_empty = relation in self.NEVER_EMPTY_RELATIONS

        instruction = NEVER_EMPTY_INSTRUCTION if (never_empty or force_answer) else STANDARD_INSTRUCTION
        if force_answer:
            instruction = f"{instruction} {FORCE_ANSWER_SUFFIX}"
        if relation in self.ITERATIVE_RELATIONS and not already_mentioned:
            instruction = f"{instruction} {MANY_WINNERS_HINT}"
        if with_reasoning:
            instruction = f"{instruction} {REASONING_SUFFIX}"

        lines = [instruction, ""]
        demonstrated = []
        for example in self.few_shot_examples.get(relation, []):
            if (self.exclude_target_from_shots
                    and example["SubjectEntity"] == subject_entity):
                continue
            # The old path fabricated the same placeholder reason for every
            # demonstration ("this is a specific, well-documented fact"). That
            # taught exactly the generic rationalization the downstream judge
            # was instructed to reject. Only show reasoning demonstrations when
            # a real, curated Reason field exists; otherwise use the instruction
            # and candidate diversity without contradictory few-shot prose.
            if (with_reasoning and not example.get("Reason")
                    and self.reasoning_demo_policy == "require-curated-reason"):
                continue
            demonstrated.append(example["SubjectEntity"])
            example_question = template.format(subject_entity=example["SubjectEntity"])
            lines.append(f"Q: {example_question}")
            answer_line = f"A: {format_answer(example['ObjectEntities'])}"
            if with_reasoning and example.get("Reason"):
                answer_line += f"\nReason: {example['Reason']}"
            lines.append(answer_line)
            lines.append("")

        if already_mentioned:
            lines.append(f"Q: {question}")
            lines.append(
                f"A (already found: {', '.join(already_mentioned)}): List ONLY "
                "winners not mentioned above, comma-separated. If there are no "
                "more, answer \"None\"."
            )
        else:
            lines.append(f"Q: {question}")
            lines.append("A:")

        user_message = "\n".join(lines)
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _batched_generate(
        self, prompts: List[str], max_new_tokens: int, progress_label: Optional[str] = None
    ) -> List[str]:
        results: List[str] = []
        total = len(prompts)
        for start in range(0, total, self.batch_size):
            batch_prompts = prompts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch_prompts, return_tensors="pt", padding=True
            ).to(self.model.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            results.extend(decoded)
            if progress_label:
                logger.info(f"{progress_label}: {min(start + self.batch_size, total)}/{total}")
        return results

    def _sampled_generate(self, prompt: str, max_new_tokens: int, n_samples: int, temperature: float) -> List[str]:
        """N independent stochastic samples of the SAME prompt, batched into
        one generate() call (tiled n_samples times) rather than n_samples
        sequential calls. Measured on this hardware: ~1.5x the wall-clock of
        a single sample at n_samples=5, not ~5x, because decoding is
        memory-bandwidth-bound -- reading the model's weights dominates
        cost, and that cost is shared across everything in the batch."""
        tiled = [prompt] * n_samples
        encoded = self.tokenizer(tiled, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    @staticmethod
    def _aggregate_numeric_samples(sample_texts: List[str]) -> List[str]:
        values = []
        for text in sample_texts:
            parsed = parse_answer(text, is_numeric=True)
            if not parsed:
                continue
            try:
                values.append(float(parsed[0].replace(",", "")))
            except ValueError:
                continue
        if not values:
            return []
        values.sort()
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 == 1 else (values[mid - 1] + values[mid]) / 2
        return [str(int(median))] if median == int(median) else [str(median)]

    def _generate_numeric_self_consistent(self, subjects: List[str], relation: str) -> List[List[str]]:
        predictions = []
        sample_sets = []
        seeds = []
        for i, s in enumerate(subjects):
            seeds.append(self._set_subject_seed(s, relation, "numeric"))
            prompt = self._build_prompt(s, relation)
            samples = self._sampled_generate(
                prompt, self.max_new_tokens, self.numeric_samples, self.numeric_sample_temperature
            )
            sample_sets.append(samples)
            predictions.append(self._aggregate_numeric_samples(samples))
            logger.info(f"{relation} (self-consistency, n={self.numeric_samples}): {i + 1}/{len(subjects)} subjects done")

        # Same never-empty safety net as the plain path: these relations can
        # never legitimately be empty, but self-consistency can still land
        # on all-unparseable samples for a subject the model has no signal
        # on at all. One forceful greedy retry rather than an empty answer.
        retry_idx = [i for i, p in enumerate(predictions) if not p]
        if retry_idx:
            logger.info(f"{relation}: retrying {len(retry_idx)} subject(s) with no parseable self-consistency samples")
            retry_prompts = [self._build_prompt(subjects[i], relation, force_answer=True) for i in retry_idx]
            retry_texts = self._batched_generate(retry_prompts, self.max_new_tokens, progress_label=f"{relation} (retry)")
            for i, t in zip(retry_idx, retry_texts):
                predictions[i] = parse_answer(t, is_numeric=True)
                sample_sets[i].append(t)
        for subject, prediction, samples, seed in zip(
                subjects, predictions, sample_sets, seeds):
            self._record_raw(
                subject, relation, prediction,
                strategy="numeric-self-consistency", raw_samples=samples,
                generation_seed=seed)
        return predictions

    def _judge(self, question: str, candidates: List[str], is_list: bool) -> str:
        instruction_template = JUDGE_LIST_INSTRUCTION_TEMPLATE if is_list else JUDGE_SINGLE_INSTRUCTION_TEMPLATE
        lines = [instruction_template.format(n=len(candidates)), "", f"Question: {question}", ""]
        for i, candidate in enumerate(candidates, 1):
            # Candidates are raw multi-line "answer\nReason: ..." text --
            # flatten to keep each one a single, clearly-delimited line.
            flattened = " ".join(candidate.strip().split())
            lines.append(f"Candidate {i}: {flattened}")
        lines.append("")
        lines.append("Final answer:")

        user_message = "\n".join(lines)
        messages = [{"role": "user", "content": user_message}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking,
        )
        max_new_tokens = self.judge_list_max_new_tokens if is_list else self.judge_max_new_tokens
        return self._batched_generate([prompt], max_new_tokens)[0]

    def _generate_with_judge(self, subjects: List[str], relation: str) -> List[List[str]]:
        is_numeric = relation in self.NUMERIC_RELATIONS
        predictions = []
        for i, s in enumerate(subjects):
            seed = self._set_subject_seed(s, relation, "candidate-judge")
            question = self.prompt_templates[relation].format(subject_entity=s)
            candidate_prompt = self._build_prompt(s, relation, with_reasoning=True)
            candidates = self._sampled_generate(
                candidate_prompt, self.judge_candidate_max_new_tokens,
                self.judge_n_samples, self.judge_sample_temperature,
            )
            final_text = self._judge(question, candidates, is_list=False)
            prediction = parse_answer(final_text, is_numeric)
            predictions.append(prediction)
            self._record_raw(
                s, relation, prediction, strategy="candidate-judge",
                raw_samples=candidates, judge_output=final_text,
                generation_seed=seed)
            logger.info(f"{relation} (judge, n={self.judge_n_samples}): {i + 1}/{len(subjects)} subjects done")
        return predictions

    @staticmethod
    def _drop_self_reference(subject_entity: str, items: List[str]) -> List[str]:
        subj_norm = subject_entity.strip().lower()
        return [item for item in items if item.strip().lower() != subj_norm]

    @staticmethod
    def _dedup_new(items: List[str], seen: set) -> List[str]:
        fresh = []
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                fresh.append(item)
        return fresh

    def _generate_standard(self, subjects: List[str], relation: str) -> List[List[str]]:
        is_numeric = relation in self.NUMERIC_RELATIONS
        never_empty = relation in self.NEVER_EMPTY_RELATIONS

        prompts = [self._build_prompt(s, relation) for s in subjects]
        texts = self._batched_generate(prompts, self.max_new_tokens, progress_label=relation)
        predictions = [parse_answer(t, is_numeric) for t in texts]
        retry_text_by_index = {}

        # Safety net: this relation can never legitimately be empty, but the
        # model abstained anyway -- give it one forceful second attempt
        # rather than accepting a guaranteed-wrong empty prediction.
        if never_empty:
            retry_idx = [i for i, p in enumerate(predictions) if not p]
            if retry_idx:
                logger.info(f"{relation}: retrying {len(retry_idx)} abstained subject(s)")
                retry_prompts = [
                    self._build_prompt(subjects[i], relation, force_answer=True)
                    for i in retry_idx
                ]
                retry_texts = self._batched_generate(
                    retry_prompts, self.max_new_tokens, progress_label=f"{relation} (retry)"
                )
                for i, t in zip(retry_idx, retry_texts):
                    predictions[i] = parse_answer(t, is_numeric)
                    retry_text_by_index[i] = t

        for i, (subject, prediction, text) in enumerate(
                zip(subjects, predictions, texts)):
            self._record_raw(
                subject, relation, prediction, strategy="standard",
                raw_samples=[text], retry_output=retry_text_by_index.get(i))

        return predictions

    def _generate_iterative(self, subject_entity: str, relation: str) -> List[str]:
        collected: List[str] = []
        seen: set = set()
        rounds = []
        question = self.prompt_templates[relation].format(subject_entity=subject_entity)

        for round_idx in range(self.award_max_rounds):
            seed = self._set_subject_seed(
                subject_entity, relation, f"iterative-{round_idx}")
            candidate_prompt = self._build_prompt(
                subject_entity,
                relation,
                force_answer=(round_idx == 0),
                already_mentioned=collected or None,
                with_reasoning=True,
            )
            candidates = self._sampled_generate(
                candidate_prompt, self.judge_list_candidate_max_new_tokens,
                self.judge_n_samples, self.judge_sample_temperature,
            )
            judged_text = self._judge(question, candidates, is_list=True)
            fresh = self._dedup_new(parse_answer(judged_text, is_numeric=False), seen)
            rounds.append({
                "round": round_idx + 1,
                "generation_seed": seed,
                "raw_samples": candidates,
                "judge_output": judged_text,
                "new_items": list(fresh),
            })
            logger.info(
                f"{relation} `{subject_entity}`: round {round_idx + 1}/"
                f"{self.award_max_rounds} (judge, n={self.judge_n_samples}) -> "
                f"{len(fresh)} new ({len(collected) + len(fresh)} total)"
            )
            if not fresh:
                break
            collected.extend(fresh)

        # Never-empty safety net: awardWonBy has no legitimate empty answer, but
        # the iterative judge can filter every candidate away and leave nothing.
        # An empty list there is a guaranteed-wrong prediction, so force one
        # direct answer instead (mirrors the retry in the standard/numeric
        # paths). This is the standalone bug the FABLE handoff flagged as open.
        if not collected and relation in self.NEVER_EMPTY_RELATIONS:
            logger.info(f"{relation} `{subject_entity}`: iterative judge returned "
                        "empty; forcing one direct answer (never-empty relation)")
            forced_prompt = self._build_prompt(
                subject_entity, relation, force_answer=True, with_reasoning=False)
            forced_text = self._batched_generate(
                [forced_prompt], self.judge_list_candidate_max_new_tokens)[0]
            collected = self._dedup_new(
                parse_answer(forced_text, is_numeric=False), seen)
            rounds.append({"round": "forced_fallback",
                           "raw_samples": [forced_text],
                           "new_items": list(collected)})

        self._record_raw(
            subject_entity, relation, collected, strategy="iterative-judge",
            rounds=rounds)
        return collected

    def generate_predictions(self, inputs: List[Dict[str, str]]) -> List[List[str]]:
        self.raw_records = []
        # Group by relation so each batch shares one prompt style and one
        # max_new_tokens budget (awardWonBy needs far more tokens than the
        # rest), instead of mixing relations inside arbitrary fixed batches.
        indices_by_relation: Dict[str, List[int]] = defaultdict(list)
        for idx, item in enumerate(inputs):
            indices_by_relation[item["Relation"]].append(idx)

        predictions: List[Optional[List[str]]] = [None] * len(inputs)

        for relation, idxs in indices_by_relation.items():
            subjects = [inputs[i]["SubjectEntity"] for i in idxs]

            if relation in self.ITERATIVE_RELATIONS:
                rel_predictions = []
                for i, s in enumerate(subjects):
                    rel_predictions.append(self._generate_iterative(s, relation))
                    logger.info(f"{relation}: {i + 1}/{len(subjects)} subjects done")
            elif relation in self.NUMERIC_SELF_CONSISTENCY_RELATIONS:
                rel_predictions = self._generate_numeric_self_consistent(subjects, relation)
            elif relation in self.SINGLE_SHOT_JUDGE_RELATIONS:
                rel_predictions = self._generate_with_judge(subjects, relation)
            else:
                rel_predictions = self._generate_standard(subjects, relation)

            # The model sometimes lists the subject itself as one of its own
            # objects (e.g. a country "bordering" itself) -- always wrong,
            # never worth the tokens it costs to generate. Cheap, zero-risk
            # filter: measured +0.013 macro-f1 on countryLandBordersCountry
            # alone (9/68 subjects were affected).
            rel_predictions = [
                self._drop_self_reference(s, preds)
                for s, preds in zip(subjects, rel_predictions)
            ]

            for i, pred in zip(idxs, rel_predictions):
                predictions[i] = pred

            logger.info(f"Finished relation `{relation}` ({len(idxs)} subjects)")

        # Ensure trace records expose the exact final prediction after global
        # post-processing (not the pre-self-reference-filter intermediate).
        final_by_key = {
            (item["SubjectEntity"], item["Relation"]): prediction
            for item, prediction in zip(inputs, predictions)
        }
        for record in self.raw_records:
            record["ObjectEntities"] = final_by_key[
                (record["SubjectEntity"], record["Relation"])]
        return predictions
