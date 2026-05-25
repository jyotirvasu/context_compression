"""
AutoPrompt Trigger Search Algorithm.
The main optimization loop that searches for optimal trigger tokens.
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from .gradient_search import GradientStorage, hotflip_attack, get_loss, get_embeddings
from .template import TriggerTemplatizer, add_task_specific_tokens, encode_label
from .data import Collator, load_trigger_dataset

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for AutoPrompt trigger search."""
    # Model
    model_name: str = "bert-base-uncased"
    seed: int = 42

    # Template
    template: str = "[T] [T] [T] {sentence} [P] ."
    label_map: Optional[Dict[str, str]] = None
    label_field: str = "label"
    tokenize_labels: bool = False
    use_ctx: bool = False

    # Search hyperparameters
    num_candidates: int = 10          # Candidates per position per iteration
    accumulation_steps: int = 10      # Gradient accumulation steps
    iters: int = 50                   # Number of search iterations
    patience: int = 5                 # Early stopping patience

    # Data
    batch_size: int = 32
    eval_size: int = 256
    limit: Optional[int] = None

    # Filtering
    filter_special: bool = True       # Filter special tokens from candidates
    filter_labels: bool = True        # Filter label tokens from candidates
    filter_capitalized: bool = False  # Filter capitalized words (proper nouns)

    # Initial trigger
    initial_trigger: Optional[List[str]] = None

    # Device
    device: str = "auto"  # "auto", "cuda", or "cpu"


class PredictWrapper:
    """
    Wraps a HuggingFace LM model for trigger token experiments.

    Handles:
    - Replacing trigger mask positions with current trigger IDs
    - Extracting logits only at prediction (masked) positions
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs: Dict, trigger_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with trigger tokens injected.

        Args:
            model_inputs: Dict with input_ids, attention_mask, trigger_mask, predict_mask
            trigger_ids: Current trigger token IDs [1, num_triggers]

        Returns:
            predict_logits: Logits at prediction positions [batch, vocab_size]
        """
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')

        # Replace trigger placeholders with current trigger tokens
        model_inputs = self._replace_triggers(model_inputs, trigger_ids, trigger_mask)

        # Forward pass
        logits = self._model(**model_inputs)[0]

        # Extract logits at prediction positions only
        predict_logits = logits.masked_select(
            predict_mask.unsqueeze(-1)
        ).view(logits.size(0), -1)

        return predict_logits

    @staticmethod
    def _replace_triggers(model_inputs, trigger_ids, trigger_mask):
        """Replace trigger mask positions with actual trigger IDs."""
        out = model_inputs.copy()
        input_ids = model_inputs['input_ids']
        trigger_ids_expanded = trigger_ids.repeat(trigger_mask.size(0), 1)
        try:
            filled = input_ids.masked_scatter(trigger_mask, trigger_ids_expanded)
        except RuntimeError:
            filled = input_ids
        out['input_ids'] = filled
        return out


class AccuracyFn:
    """
    Evaluation function that computes accuracy for multi-token label prediction.

    Compares target label log-probability against all other label log-probabilities.
    A prediction is correct if the target label has the highest log-probability.
    """

    def __init__(self, tokenizer, label_map: Dict[str, str], device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(
                encode_label(tokenizer, label_tokens, tokenize_labels).to(device)
            )
            self._pred_to_label.append(label)
        logger.info(f"AccuracyFn labels: {list(label_map.keys())}")

    def __call__(self, predict_logits: torch.Tensor, gold_label_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-instance accuracy.

        Args:
            predict_logits: Model logits at prediction position [batch, vocab_size]
            gold_label_ids: Gold label token IDs [batch, num_label_tokens]

        Returns:
            Tensor of 0/1 accuracy values [batch]
        """
        gold_logp = get_loss(predict_logits, gold_label_ids)
        bsz = predict_logits.size(0)

        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)

        # Count how many labels have logp <= gold (lower loss = better)
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # Correct if at most 1 label (itself) is <= gold

        return correct.float()

    def predict(self, predict_logits: torch.Tensor) -> List[str]:
        """Return predicted label strings."""
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        return [self._pred_to_label[x] for x in predictions.tolist()]


class AutoPromptSearcher:
    """
    Main AutoPrompt trigger search algorithm.

    Implements the full pipeline:
    1. Load pretrained MLM
    2. Process data through template
    3. Iteratively optimize trigger tokens via gradient-guided search
    4. Return best trigger tokens found

    Usage:
        config = SearchConfig(
            model_name="bert-base-uncased",
            template="[T] [T] [T] {sentence} [P] .",
            label_map={"positive": "good", "negative": "bad"},
        )
        searcher = AutoPromptSearcher(config)
        result = searcher.search(train_path, dev_path)
        print(f"Best triggers: {result['best_tokens']}")
        print(f"Dev accuracy: {result['best_dev_metric']:.4f}")
    """

    def __init__(self, config: SearchConfig):
        self.config = config
        self._setup_device()
        self._load_model()

    def _setup_device(self):
        """Configure computation device."""
        if self.config.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)
        logger.info(f"Using device: {self.device}")

    def _load_model(self):
        """Load pretrained model, tokenizer, and setup gradient storage."""
        logger.info(f"Loading model: {self.config.model_name}")
        self.model_config = AutoConfig.from_pretrained(self.config.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, add_prefix_space=True
        )
        add_task_specific_tokens(self.tokenizer)

        # Setup gradient capture on embedding layer
        self.embeddings = get_embeddings(self.model, self.model_config)
        self.embedding_gradient = GradientStorage(self.embeddings)

        self.predictor = PredictWrapper(self.model)
        logger.info("Model loaded successfully")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)

    def _build_filter(self, label_map, train_dataset) -> torch.Tensor:
        """
        Build a filter tensor to exclude certain tokens from candidate selection.

        Filtered tokens get a large negative bias so they won't be selected
        by the hotflip attack. This includes:
        - Special tokens ([CLS], [SEP], [PAD], etc.)
        - Label tokens (to prevent trivial solutions)
        - Capitalized words (optional, removes proper nouns)
        """
        token_filter = torch.zeros(
            self.tokenizer.vocab_size, dtype=torch.float32, device=self.device
        )

        if self.config.filter_labels and label_map:
            logger.info("Filtering label tokens from candidates")
            for label_tokens in label_map.values():
                label_ids = encode_label(self.tokenizer, label_tokens).unsqueeze(0)
                token_filter[label_ids] = -1e32

        if self.config.filter_special:
            logger.info("Filtering special tokens")
            for idx in self.tokenizer.all_special_ids:
                token_filter[idx] = -1e32

            if self.config.filter_capitalized:
                for word, idx in self.tokenizer.get_vocab().items():
                    if len(word) <= 1 or idx >= self.tokenizer.vocab_size:
                        continue
                    # Filter capitalized words (lazy proper noun removal)
                    decoded = self.tokenizer.decode([idx])
                    if decoded and decoded[0].isupper():
                        token_filter[idx] = -1e32

        return token_filter

    def search(self, train_path, dev_path) -> Dict:
        """
        Run the full AutoPrompt trigger search.

        This is the main entry point. It:
        1. Loads data and creates templatizer
        2. Initializes trigger tokens
        3. Iteratively optimizes via gradient-guided search
        4. Returns best triggers and metrics

        Args:
            train_path: Path to training data
            dev_path: Path to dev/validation data

        Returns:
            Dict with:
                - best_tokens: List of best trigger token strings
                - best_trigger_ids: Tensor of best trigger token IDs
                - best_dev_metric: Best dev set metric achieved
                - history: List of (iteration, dev_metric) tuples
                - elapsed_time: Total search time in seconds
        """
        self._set_seed()
        cfg = self.config

        # Setup templatizer
        templatizer = TriggerTemplatizer(
            template=cfg.template,
            config=self.model_config,
            tokenizer=self.tokenizer,
            label_map=cfg.label_map,
            label_field=cfg.label_field,
            tokenize_labels=cfg.tokenize_labels,
            add_special_tokens=False,
            use_ctx=cfg.use_ctx,
        )

        # Initialize trigger tokens
        if cfg.initial_trigger:
            trigger_ids = self.tokenizer.convert_tokens_to_ids(cfg.initial_trigger)
            assert len(trigger_ids) == templatizer.num_trigger_tokens
        else:
            trigger_ids = [self.tokenizer.mask_token_id] * templatizer.num_trigger_tokens

        trigger_ids = torch.tensor(trigger_ids, device=self.device).unsqueeze(0)
        best_trigger_ids = trigger_ids.clone()

        # Setup evaluation function
        if cfg.label_map:
            evaluation_fn = AccuracyFn(self.tokenizer, cfg.label_map, self.device)
        else:
            evaluation_fn = lambda x, y: -get_loss(x, y)

        # Load data
        logger.info("Loading datasets...")
        collator = Collator(pad_token_id=self.tokenizer.pad_token_id)

        train_dataset = load_trigger_dataset(
            train_path, templatizer, use_ctx=cfg.use_ctx, limit=cfg.limit
        )
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator
        )

        dev_dataset = load_trigger_dataset(dev_path, templatizer, use_ctx=cfg.use_ctx)
        dev_loader = DataLoader(
            dev_dataset, batch_size=cfg.eval_size, shuffle=False, collate_fn=collator
        )

        logger.info(f"Train: {len(train_dataset)} instances, Dev: {len(dev_dataset)} instances")

        # Build token filter
        token_filter = self._build_filter(cfg.label_map, train_dataset)

        # Initial evaluation
        dev_metric = self._evaluate(dev_loader, trigger_ids, evaluation_fn)
        logger.info(f"Initial dev metric: {dev_metric:.4f}")

        best_dev_metric = -float('inf')
        history = [(0, dev_metric)]
        patience_counter = 0

        # ========== Main Search Loop ==========
        start_time = time.time()

        for iteration in range(1, cfg.iters + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{cfg.iters}")

            # Step 1: Accumulate gradients
            averaged_grad = self._accumulate_gradients(
                train_loader, trigger_ids, templatizer.num_trigger_tokens
            )

            # Step 2: Select a random trigger position to optimize
            token_to_flip = random.randrange(templatizer.num_trigger_tokens)

            # Step 3: HotFlip attack - find candidate replacements
            candidates = hotflip_attack(
                averaged_grad[token_to_flip],
                self.embeddings.weight,
                increase_loss=False,
                num_candidates=cfg.num_candidates,
                token_filter=token_filter,
            )

            # Step 4: Evaluate candidates
            best_candidate_idx = self._evaluate_candidates(
                train_loader, trigger_ids, candidates, token_to_flip, evaluation_fn
            )

            if best_candidate_idx is not None:
                # Found improvement - update trigger
                trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                current_tokens = self.tokenizer.convert_ids_to_tokens(
                    trigger_ids.squeeze(0).tolist()
                )
                logger.info(f"Updated trigger position {token_to_flip}: {current_tokens}")
            else:
                logger.info("No improvement found. Skipping evaluation.")
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(f"Early stopping after {cfg.patience} iterations without improvement")
                    break
                continue

            # Step 5: Evaluate on dev set
            dev_metric = self._evaluate(dev_loader, trigger_ids, evaluation_fn)
            history.append((iteration, dev_metric))

            trigger_tokens = self.tokenizer.convert_ids_to_tokens(
                trigger_ids.squeeze(0).tolist()
            )
            logger.info(f"Trigger tokens: {trigger_tokens}")
            logger.info(f"Dev metric: {dev_metric:.4f}")

            if dev_metric > best_dev_metric:
                logger.info(">>> New best! <<<")
                best_dev_metric = dev_metric
                best_trigger_ids = trigger_ids.clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(f"Early stopping after {cfg.patience} iterations without improvement")
                    break

        elapsed = time.time() - start_time

        # Final results
        best_tokens = self.tokenizer.convert_ids_to_tokens(
            best_trigger_ids.squeeze(0).tolist()
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"Search complete in {elapsed:.1f}s")
        logger.info(f"Best trigger tokens: {best_tokens}")
        logger.info(f"Best dev metric: {best_dev_metric:.4f}")

        return {
            'best_tokens': best_tokens,
            'best_trigger_ids': best_trigger_ids,
            'best_dev_metric': best_dev_metric,
            'history': history,
            'elapsed_time': elapsed,
        }

    def _accumulate_gradients(
        self, train_loader, trigger_ids, num_trigger_tokens
    ) -> torch.Tensor:
        """
        Accumulate gradients over multiple batches.

        The averaged gradient indicates which direction in embedding space
        would decrease the loss the most for each trigger position.

        Returns:
            averaged_grad: [num_trigger_tokens, emb_dim]
        """
        self.model.zero_grad()
        train_iter = iter(train_loader)
        averaged_grad = None

        for step in range(self.config.accumulation_steps):
            try:
                model_inputs, labels = next(train_iter)
            except StopIteration:
                logger.warning("Ran out of training data during accumulation")
                break

            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            labels = labels.to(self.device)

            # Forward + backward
            predict_logits = self.predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            # Extract gradient at trigger positions
            grad = self.embedding_gradient.get()
            bsz, _, emb_dim = grad.size()

            # Mask to only trigger positions
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, num_trigger_tokens, emb_dim)

            # Accumulate averaged gradient
            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / self.config.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / self.config.accumulation_steps

        return averaged_grad

    def _evaluate_candidates(
        self, train_loader, trigger_ids, candidates, token_to_flip, evaluation_fn
    ) -> Optional[int]:
        """
        Evaluate candidate trigger tokens and return the best one.

        For each candidate, temporarily replace the trigger token and measure
        performance. If any candidate beats the current trigger, return its index.

        Returns:
            Index of best candidate, or None if no improvement
        """
        train_iter = iter(train_loader)
        current_score = 0
        candidate_scores = torch.zeros(len(candidates), device=self.device)
        denom = 0

        for step in range(self.config.accumulation_steps):
            try:
                model_inputs, labels = next(train_iter)
            except StopIteration:
                break

            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            labels = labels.to(self.device)

            with torch.no_grad():
                # Score current trigger
                predict_logits = self.predictor(model_inputs, trigger_ids)
                eval_metric = evaluation_fn(predict_logits, labels)
                current_score += eval_metric.sum()
                denom += labels.size(0)

                # Score each candidate
                for i, candidate in enumerate(candidates):
                    temp_trigger = trigger_ids.clone()
                    temp_trigger[:, token_to_flip] = candidate
                    predict_logits = self.predictor(model_inputs, temp_trigger)
                    eval_metric = evaluation_fn(predict_logits, labels)
                    candidate_scores[i] += eval_metric.sum()

        # Check if any candidate improves over current
        if (candidate_scores > current_score).any():
            best_idx = candidate_scores.argmax().item()
            best_score = candidate_scores[best_idx]
            logger.info(
                f"Better trigger found! Score: {best_score / (denom + 1e-13):.4f} "
                f"(was {current_score / (denom + 1e-13):.4f})"
            )
            return best_idx
        return None

    def _evaluate(self, dev_loader, trigger_ids, evaluation_fn) -> float:
        """Evaluate current triggers on dev set."""
        numerator = 0
        denominator = 0

        for model_inputs, labels in dev_loader:
            model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}
            labels = labels.to(self.device)
            with torch.no_grad():
                predict_logits = self.predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)

        return numerator / (denominator + 1e-13)
