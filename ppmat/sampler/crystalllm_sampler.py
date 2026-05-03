# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CrystalLLM Sampler: autoregressive and MCTS-guided crystal structure generation.

Ported from lantunes/CrystaLLM (MIT License).
Reference: Antunes et al., Nature Communications, 2024.
DOI: 10.1038/s41467-024-54639-7

Provides two sampling modes:
1. Standard autoregressive sampling (temperature + top-k)
2. MCTS-guided sampling with pluggable evaluation functions
"""

import math
import os
import random
import traceback
from math import log, sqrt
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F

from ppmat.metrics.crystal_metrics import (
    bond_length_reasonableness_score,
    extract_numeric_property,
    extract_space_group_symbol,
    get_unit_cell_volume,
    is_atom_site_multiplicity_consistent,
    is_formula_consistent,
    is_space_group_consistent,
    remove_atom_props_block,
    replace_symmetry_operators,
)
from ppmat.models.crystalllm import CIFTokenizer, CrystalLLM, GPTConfig


# ---------------------------------------------------------------------------
# MCTS Evaluator
# ---------------------------------------------------------------------------


class MCTSEvaluator:
    """Evaluates generated CIF token sequences for MCTS reward computation.

    Uses crystal validity metrics and an optional external scorer. If no
    external scorer is provided, reward is based solely on validity.

    Args:
        tokenizer: CIFTokenizer instance.
        scorer: Optional callable(cif_str) -> float. External scoring function
            (e.g., M3GNet energy predictor). If None, valid structures get +1.0.
        bond_length_acceptability_cutoff: Minimum bond score for validity.
        reward_k: Sensitivity of reward to score deviations from mean.
        out_dir: Optional directory to save generated CIF files.
    """

    def __init__(
        self,
        tokenizer: CIFTokenizer,
        scorer: Optional[Callable] = None,
        bond_length_acceptability_cutoff: float = 1.0,
        reward_k: float = 2.0,
        out_dir: Optional[str] = None,
    ):
        self._scorer = scorer
        self._tokenizer = tokenizer
        self._bond_length_acceptability_cutoff = bond_length_acceptability_cutoff
        self._k = reward_k
        self._out_dir = out_dir
        self._num_valid = 0
        self._all_scores: List[float] = []
        self._all_cifs: List[str] = []

    def _postprocess(self, cif_str: str) -> str:
        """Post-process generated CIF: validate volume, fix symmetry ops."""
        a = extract_numeric_property(cif_str, "_cell_length_a")
        b = extract_numeric_property(cif_str, "_cell_length_b")
        c = extract_numeric_property(cif_str, "_cell_length_c")
        alpha = extract_numeric_property(cif_str, "_cell_angle_alpha")
        beta = extract_numeric_property(cif_str, "_cell_angle_beta")
        gamma = extract_numeric_property(cif_str, "_cell_angle_gamma")
        get_unit_cell_volume(a, b, c, alpha, beta, gamma)

        space_group_symbol = extract_space_group_symbol(cif_str)
        if space_group_symbol is not None and space_group_symbol != "P 1":
            cif_str = replace_symmetry_operators(cif_str, space_group_symbol)

        cif_str = remove_atom_props_block(cif_str)
        return cif_str

    def _is_valid(self, cif_str: str) -> Tuple[bool, str, Optional[float]]:
        """Check CIF validity (formula, multiplicity, bonds, space group)."""
        if not is_formula_consistent(cif_str):
            return False, "inconsistent composition", None
        if not is_atom_site_multiplicity_consistent(cif_str):
            return False, "inconsistent atom site multiplicity", None
        bond_score = bond_length_reasonableness_score(cif_str)
        if bond_score < self._bond_length_acceptability_cutoff:
            return (
                False,
                f"unreasonable bonds ({(1 - bond_score) * 100:.0f}%)",
                bond_score,
            )
        sg = extract_space_group_symbol(cif_str)
        if sg is not None and not is_space_group_consistent(cif_str, sg):
            return False, "inconsistent space group", None
        return True, "", None

    def _get_reward(self, score: float) -> float:
        """Map score to [0, 1] reward using running z-score normalization."""
        self._all_scores.append(score)
        if len(self._all_scores) == 1 or len(np.unique(self._all_scores)) == 1:
            return 0.5
        mu = np.mean(self._all_scores)
        sigma = np.std(self._all_scores)
        return 1 / (1 + math.e ** (self._k * ((score - mu) / sigma)))

    def _write_cif(self, cif: str, score: float, reward: float, cif_id: int, iter_num: int):
        """Write generated CIF to file and update CSV log."""
        if self._out_dir is None:
            return
        os.makedirs(self._out_dir, exist_ok=True)
        cif_file = f"generated_{cif_id}.cif"
        cif_path = os.path.join(self._out_dir, cif_file)
        if not os.path.exists(cif_path):
            with open(cif_path, "wt") as f:
                f.write(cif)
            csv_path = os.path.join(self._out_dir, "results.csv")
            if not os.path.exists(csv_path):
                with open(csv_path, "wt") as f:
                    f.write("file,iteration,score,reward\n")
            with open(csv_path, "a") as f:
                f.write(f"{cif_file},{iter_num},{score},{reward}\n")

    def __call__(self, token_sequence: List[int], iter_num: int) -> float:
        """Evaluate a generated token sequence, returning a reward in [-1, 1]."""
        cif = self._tokenizer.decode(token_sequence)
        try:
            cif = self._postprocess(cif)
            valid, msg, bond_score = self._is_valid(cif)
            if not valid:
                if bond_score is not None:
                    return -(1 - bond_score)
                return -1.0
        except Exception:
            return -1.0

        self._num_valid += 1

        if self._scorer is not None:
            try:
                score = self._scorer(cif)
            except Exception:
                return -1.0
            if math.isnan(score):
                return -1.0
            reward = self._get_reward(score)
        else:
            # No external scorer: valid structures get a fixed positive reward
            reward = 0.8

        self._write_cif(cif, score if self._scorer else 1.0, reward, self._num_valid, iter_num)
        self._all_cifs.append(cif)
        return reward


# ---------------------------------------------------------------------------
# MCTS Language Model wrapper (Paddle)
# ---------------------------------------------------------------------------


class MCTSLanguageModel:
    """Wraps CrystalLLM for MCTS rollout and child probability computation."""

    def __init__(
        self,
        model: CrystalLLM,
        config: GPTConfig,
        child_ids: List[int],
        device: str,
        temperature: float,
    ):
        self._model = model
        self._model.eval()
        self._config = config
        self._child_ids = child_ids
        self._device = device
        self._temperature = temperature

    def rollout(
        self, rollout_state: List[int], width: int, max_depth: int, newline_id: int
    ) -> List[int]:
        """Perform a rollout from the given state using temperature sampling."""
        idx = paddle.to_tensor(
            [rollout_state], dtype="int64", place=paddle.CPUPlace()
            if self._device == "cpu" else None
        )
        prev_id = None
        for _ in range(max_depth):
            idx_cond = (
                idx if idx.shape[1] <= self._config.block_size
                else idx[:, -self._config.block_size :]
            )
            logits = self._model._forward(idx_cond)
            logits = logits[:, -1, :] / self._temperature
            if width is not None:
                k = min(width, logits.shape[-1])
                v, _ = paddle.topk(logits, k)
                logits = paddle.where(
                    logits < v[:, -1:],
                    paddle.full_like(logits, float("-inf")),
                    logits,
                )
            probs = F.softmax(logits, axis=-1)
            idx_next = paddle.multinomial(probs, num_samples=1)
            idx = paddle.concat([idx, idx_next], axis=1)

            cur_id = idx_next.item()
            if prev_id is not None and prev_id == newline_id and cur_id == newline_id:
                break
            prev_id = cur_id
        return idx[0].tolist()

    def top_n_vocab_with_weights(
        self, n: int, token_sequence: List[int]
    ) -> Tuple[List[int], List[float]]:
        """Get top-n tokens and their normalized probabilities for the next position."""
        idx = paddle.to_tensor(
            [token_sequence], dtype="int64", place=paddle.CPUPlace()
            if self._device == "cpu" else None
        )
        idx_cond = (
            idx if idx.shape[1] <= self._config.block_size
            else idx[:, -self._config.block_size :]
        )
        logits = self._model._forward(idx_cond)
        logits = logits[:, -1, :] / self._temperature

        log_probs = F.log_softmax(logits, axis=-1).squeeze(0)

        tokens_and_log_probs = []
        for child_id in self._child_ids:
            lp = log_probs[child_id].item()
            tokens_and_log_probs.append((child_id, lp))

        top_n = sorted(tokens_and_log_probs, key=lambda k: k[1], reverse=True)[:n]
        top_n_child_ids = [t[0] for t in top_n]
        top_n_weights = self._normalize([t[1] for t in top_n])
        return top_n_child_ids, top_n_weights

    @staticmethod
    def _normalize(log_probs: List[float]) -> List[float]:
        probs = [math.exp(lp) for lp in log_probs]
        total = sum(probs)
        return [p / total for p in probs]


# ---------------------------------------------------------------------------
# MCTS Tree Components
# ---------------------------------------------------------------------------


class MCTSNode:
    """A node in the MCTS search tree."""

    def __init__(
        self,
        state: List[int],
        language_model: MCTSLanguageModel,
        width: int,
        max_depth: int,
        newline_id: int,
        parent: Optional["MCTSNode"] = None,
        tree_builder: Optional["ContextSensitiveTreeBuilder"] = None,
    ):
        self.state = state
        self._newline_id = newline_id
        self._lm = language_model
        self._width = width
        self._max_depth = max_depth
        self.wins = 0.0
        self.visits = 0.0
        self.prob = None
        self.parent = parent
        self.tree_builder = tree_builder
        self.children: List["MCTSNode"] = []
        self.untried_moves, self.child_weight_map = self._get_child_states()

    @staticmethod
    def is_complete(state: List[int], newline_id: int) -> bool:
        return len(state) > 1 and state[-2:] == [newline_id, newline_id]

    def _get_child_states(self):
        child_states = []
        child_state_weight_map = {}
        if len(self.state) < self._max_depth and not self.is_complete(
            self.state, self._newline_id
        ):
            top_ids, top_w = self._lm.top_n_vocab_with_weights(self._width, self.state)
            if self.tree_builder is not None:
                top_ids, top_w = self.tree_builder.get_child_ids_and_weights(
                    self.state, top_ids, top_w, self._lm, self._width, self._newline_id
                )
            for i in range(len(top_ids)):
                cs = (
                    self.state + top_ids[i]
                    if isinstance(top_ids[i], list)
                    else self.state + [top_ids[i]]
                )
                child_states.append(cs)
                child_state_weight_map[tuple(cs)] = top_w[i]
        return child_states, child_state_weight_map

    def has_untried_moves(self) -> bool:
        return len(self.untried_moves) > 0

    def select_untried_move(self) -> List[int]:
        return random.choice(self.untried_moves)

    def add_child(self, child_state, language_model, width, max_depth, newline_id):
        child = MCTSNode(
            child_state, language_model, width, max_depth, newline_id,
            parent=self, tree_builder=self.tree_builder,
        )
        child.prob = self.child_weight_map[tuple(child_state)]
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self) -> bool:
        return len(self.children) > 0


class ContextSensitiveTreeBuilder:
    """Handles context-dependent branching (space groups, only-child bypass)."""

    def __init__(
        self,
        tokenizer: CIFTokenizer,
        top_child_weight_cutoff: float = 0.99,
        n_space_groups: int = 0,
        bypass_only_child: bool = False,
    ):
        self._tok = tokenizer
        self._top_child_weight_cutoff = top_child_weight_cutoff
        self._n_space_groups = n_space_groups
        self._bypass_only_child = bypass_only_child

    def get_child_ids_and_weights(
        self,
        state: List[int],
        top_n_child_ids: List[int],
        top_n_weights: List[float],
        lm: MCTSLanguageModel,
        width: int,
        newline_id: int,
    ) -> Tuple[Union[List[int], List[List[int]]], List[float]]:
        tok2id = self._tok.token_to_id

        # Special handling for space group position
        if (
            len(state) > 1
            and state[-2:] == [tok2id["_symmetry_space_group_name_H-M"], tok2id[" "]]
            and self._n_space_groups > 0
        ):
            return lm.top_n_vocab_with_weights(self._n_space_groups, state)

        top_child_id = top_n_child_ids[0]
        top_child_weight = top_n_weights[0]

        if top_child_weight > self._top_child_weight_cutoff:
            if self._bypass_only_child:
                only_children = []
                while top_child_weight > self._top_child_weight_cutoff:
                    only_children.append(top_child_id)
                    new_state = state + only_children
                    if MCTSNode.is_complete(new_state, newline_id):
                        return [only_children], [1.0]
                    top_n_child_ids, top_n_weights = lm.top_n_vocab_with_weights(
                        width, new_state
                    )
                    top_child_id = top_n_child_ids[0]
                    top_child_weight = top_n_weights[0]
                extended = [only_children + [cid] for cid in top_n_child_ids]
                return extended, top_n_weights
            return [top_child_id], [1.0]

        return top_n_child_ids, top_n_weights


# ---------------------------------------------------------------------------
# Node Selectors
# ---------------------------------------------------------------------------


class MCTSNodeSelector:
    """Base class for MCTS node selection strategies."""

    def select_node(self, nodes: List[MCTSNode]) -> MCTSNode:
        raise NotImplementedError


class PUCTSelector(MCTSNodeSelector):
    """Predictor + Upper Confidence bounds applied to Trees (AlphaGo-style)."""

    def __init__(self, cpuct: float):
        self._cpuct = cpuct

    def select_node(self, nodes: List[MCTSNode]) -> MCTSNode:
        best_score, best_node = -math.inf, None
        for node in nodes:
            score = self._puct(node)
            if score > best_score:
                best_score, best_node = score, node
        return best_node

    def _puct(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return math.inf
        if node.prob is None:
            raise ValueError(f"Node has no action probability: {node.state}")
        return (
            node.wins / node.visits
            + self._cpuct * node.prob * sqrt(node.parent.visits) / (1 + node.visits)
        )


class UCTSelector(MCTSNodeSelector):
    """Upper Confidence bounds applied to Trees (classic UCB1)."""

    def __init__(self, c: float):
        self._c = c

    def select_node(self, nodes: List[MCTSNode]) -> MCTSNode:
        best_score, best_node = -math.inf, None
        for node in nodes:
            score = self._uct(node)
            if score > best_score:
                best_score, best_node = score, node
        return best_node

    def _uct(self, node: MCTSNode) -> float:
        if node.visits == 0:
            return math.inf
        if node.prob is None:
            raise ValueError(f"Node has no action probability: {node.state}")
        return (node.wins / node.visits) + self._c * sqrt(
            log(node.parent.visits) / node.visits
        )


class GreedySelector(MCTSNodeSelector):
    """Epsilon-greedy node selection."""

    def __init__(self, epsilon: float):
        self._epsilon = epsilon

    def select_node(self, nodes: List[MCTSNode]) -> MCTSNode:
        if random.random() < self._epsilon:
            return random.choice(nodes)
        best_val, best_node = -math.inf, None
        for node in nodes:
            val = node.wins / node.visits if node.visits > 0 else 0.0
            if val > best_val:
                best_val, best_node = val, node
        return best_node


# ---------------------------------------------------------------------------
# MCTS Sampler
# ---------------------------------------------------------------------------


class MCTSSampler:
    """Monte Carlo Tree Search sampler for guided crystal structure generation.

    Uses MCTS to explore the token space with an evaluation function that
    rewards valid, high-quality crystal structures.

    Args:
        model: CrystalLLM model instance.
        config: GPTConfig for the model.
        width: Number of top children to consider at each node.
        max_depth: Maximum token sequence length.
        eval_function: Callable(token_list, iter_num) -> float reward.
        node_selector: MCTSNodeSelector instance (PUCT, UCT, or Greedy).
        tokenizer: CIFTokenizer instance.
        temperature: Sampling temperature for rollouts.
        device: "cpu" or "gpu".
        tree_builder: Optional ContextSensitiveTreeBuilder.
    """

    def __init__(
        self,
        model: CrystalLLM,
        config: GPTConfig,
        width: int,
        max_depth: int,
        eval_function: Callable,
        node_selector: MCTSNodeSelector,
        tokenizer: CIFTokenizer,
        temperature: float,
        device: str,
        tree_builder: Optional[ContextSensitiveTreeBuilder] = None,
    ):
        self._width = width
        self._max_depth = max_depth
        self._eval_function = eval_function
        self._best_sequence = None
        self._node_selector = node_selector
        self._tokenizer = tokenizer
        child_ids = list(range(len(self._tokenizer.token_to_id)))
        self._lm = MCTSLanguageModel(
            model, config, child_ids=child_ids,
            temperature=temperature, device=device,
        )
        self._newline_id = self._tokenizer.token_to_id["\n"]
        self._tree_builder = tree_builder

    def search(
        self,
        start: str,
        num_simulations: int,
        stepwise: bool = False,
        n_rollouts: int = 1,
    ) -> List[int]:
        """Run MCTS search from a given prompt string.

        Args:
            start: Partial CIF text to complete.
            num_simulations: Number of MCTS simulations per step.
            stepwise: If True, return after one token expansion.
            n_rollouts: Number of rollouts per node expansion.

        Returns:
            Token sequence (list of int IDs) of the best/selected path.
        """
        state = self._tokenizer.encode(self._tokenizer.tokenize_cif(start))
        root_node = MCTSNode(
            state, self._lm, self._width, self._max_depth, self._newline_id,
            tree_builder=self._tree_builder,
        )

        if stepwise and len(root_node.untried_moves) == 1:
            return root_node.untried_moves[0]

        for iter_num in range(1, num_simulations + 1):
            node = root_node

            # Select: walk down tree using selector
            while not node.has_untried_moves() and node.has_children():
                node = self._node_selector.select_node(node.children)

            # Expand: add one child
            if node.has_untried_moves():
                move = node.select_untried_move()
                node = node.add_child(
                    move, self._lm, self._width, self._max_depth, self._newline_id
                )

            # Rollout and evaluate
            rollout_scores = []
            for _ in range(n_rollouts):
                rollout_state = self._lm.rollout(
                    node.state, self._width, self._max_depth, self._newline_id
                )
                score = self._eval_function(rollout_state, iter_num)
                self._store_best(rollout_state, score)
                rollout_scores.append(score)
            score = float(np.mean(rollout_scores))

            # Backpropagate
            while node is not None:
                node.visits += 1
                node.wins += score
                node = node.parent

        # Return the most-visited child's state
        most_visited = max(root_node.children, key=lambda c: c.visits)
        return most_visited.state

    def _store_best(self, rollout_state: List[int], score: float):
        if self._best_sequence is None or score > self._best_sequence[1]:
            self._best_sequence = (rollout_state, score)

    def get_best_sequence(self) -> Optional[Tuple[List[int], float]]:
        return self._best_sequence


# ---------------------------------------------------------------------------
# CrystalLLMSampler — unified sampler following ppmat interface
# ---------------------------------------------------------------------------


class CrystalLLMSampler:
    """Unified sampler for CrystalLLM supporting standard and MCTS modes.

    Standard mode uses temperature + top-k autoregressive sampling.
    MCTS mode uses Monte Carlo Tree Search with crystal validity evaluation.

    Args:
        model: CrystalLLM model instance (already loaded with weights).
        tokenizer: CIFTokenizer instance.
        device: "cpu" or "gpu".
        temperature: Sampling temperature (default 1.0).
        top_k: Top-k filtering (default 40, None for no filtering).
    """

    def __init__(
        self,
        model: CrystalLLM,
        tokenizer: Optional[CIFTokenizer] = None,
        device: str = "cpu",
        temperature: float = 1.0,
        top_k: Optional[int] = 40,
    ):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer or CIFTokenizer()
        self.device = device
        self.temperature = temperature
        self.top_k = top_k

    @paddle.no_grad()
    def sample(
        self,
        prompt: str,
        num_samples: int = 1,
        max_new_tokens: int = 2048,
    ) -> List[str]:
        """Generate crystal structures using standard autoregressive sampling.

        Args:
            prompt: Partial CIF text to complete (e.g., "data_" line).
            num_samples: Number of structures to generate.
            max_new_tokens: Maximum tokens to generate per sample.

        Returns:
            List of generated CIF strings.
        """
        tokens = self.tokenizer.tokenize_cif(prompt)
        ids = self.tokenizer.encode(tokens)
        newline_id = self.tokenizer.token_to_id["\n"]

        results = []
        for _ in range(num_samples):
            idx = paddle.to_tensor([ids], dtype="int64")
            generated = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                stop_token=newline_id,
            )
            cif_str = self.tokenizer.decode(generated[0].tolist())
            results.append(cif_str)
        return results

    def sample_mcts(
        self,
        prompt: str,
        num_simulations: int = 100,
        width: int = 10,
        max_depth: int = 2048,
        cpuct: float = 5.0,
        scorer: Optional[Callable] = None,
        n_space_groups: int = 0,
        bypass_only_child: bool = False,
        n_rollouts: int = 1,
        out_dir: Optional[str] = None,
    ) -> str:
        """Generate a crystal structure using MCTS-guided search.

        Args:
            prompt: Partial CIF text to complete.
            num_simulations: MCTS simulations per expansion step.
            width: Branching factor (top-k children per node).
            max_depth: Maximum token sequence length.
            cpuct: Exploration constant for PUCT selector.
            scorer: Optional callable(cif_str) -> float for external scoring.
            n_space_groups: Number of space groups to consider (0 = default).
            bypass_only_child: If True, skip deterministic single-child nodes.
            n_rollouts: Rollouts per MCTS expansion.
            out_dir: Directory to save intermediate CIF files.

        Returns:
            Generated CIF string from the best MCTS trajectory.
        """
        evaluator = MCTSEvaluator(
            tokenizer=self.tokenizer,
            scorer=scorer,
            out_dir=out_dir,
        )
        tree_builder = ContextSensitiveTreeBuilder(
            tokenizer=self.tokenizer,
            n_space_groups=n_space_groups,
            bypass_only_child=bypass_only_child,
        )
        selector = PUCTSelector(cpuct=cpuct)
        mcts = MCTSSampler(
            model=self.model,
            config=self.model.config,
            width=width,
            max_depth=max_depth,
            eval_function=evaluator,
            node_selector=selector,
            tokenizer=self.tokenizer,
            temperature=self.temperature,
            device=self.device,
            tree_builder=tree_builder,
        )

        # Stepwise MCTS: expand one token at a time
        current_prompt = prompt
        newline_id = self.tokenizer.token_to_id["\n"]
        state = self.tokenizer.encode(self.tokenizer.tokenize_cif(current_prompt))

        while len(state) < max_depth:
            state = mcts.search(
                current_prompt, num_simulations,
                stepwise=True, n_rollouts=n_rollouts,
            )
            if MCTSNode.is_complete(state, newline_id):
                break
            current_prompt = self.tokenizer.decode(state)

        # Return best sequence found across all simulations
        best = mcts.get_best_sequence()
        if best is not None:
            return self.tokenizer.decode(best[0])
        return self.tokenizer.decode(state)
