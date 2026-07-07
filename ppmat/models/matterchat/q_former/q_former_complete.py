# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Complete MatterChat model: Blip2MistralInstruct."""

import paddle
import paddle.nn as nn

from ppmat.models.matterchat.mistral.configuration_mistral import MistralConfig
from ppmat.models.matterchat.mistral.modeling_mistral import MistralForCausalLM
from ppmat.models.matterchat.q_former.q_former_base import Blip2Base


class Blip2MistralInstruct(Blip2Base):
    def __init__(
        self,
        num_query_token=32,
        prompt="test",
        max_txt_len=512,
        max_output_txt_len=1024,
        qformer_text_input=False,
        llm_tokenizer=None,
        llm_model=None,
        tokenizer_path=None,
    ):
        super().__init__()

        if llm_tokenizer is None and tokenizer_path is not None and tokenizer_path:
            from ppmat.models.matterchat.utils.tokenizer import MistralTokenizerWrapper
            import os as _os

            _resolved = _os.path.expanduser(tokenizer_path)
            llm_tokenizer = MistralTokenizerWrapper(_resolved)

        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.material_encoder = self.init_material_encoder()
        for param in self.material_encoder.parameters():
            param.stop_gradient = True

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 64)

        self.qformer_text_input = qformer_text_input
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            if llm_tokenizer is not None:
                self.Qformer.resize_token_embeddings(len(llm_tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = llm_tokenizer
        self.llm_model = (
            llm_model
            if llm_model is not None
            else MistralForCausalLM(self._get_default_mistral_config())
        )

        for param in self.llm_model.parameters():
            param.stop_gradient = True

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt

        if self.llm_tokenizer is not None:
            prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pd")
            self.prompt_length = prompt_tokens.attention_mask.sum(1).item()
        else:
            self.prompt_length = 0

    @staticmethod
    def _get_default_mistral_config():
        return MistralConfig()

    @classmethod
    def from_pretrained(cls, path):
        """Load MatterChat from a MODEL_REGISTRY-style directory.

        Directory structure:
            path/
            ├── *.yaml
            ├── tokenizer.json
            └── checkpoints/
                ├── best.pdparams  (or model.pdparams.index.json + shards)

        Args:
            path (str): Path to the extracted MODEL_REGISTRY directory.

        Returns:
            Blip2MistralInstruct: Loaded model with tokenizer and weights.
        """
        import os as _os, json as _json

        from omegaconf import OmegaConf

        yaml_files = [f for f in _os.listdir(path) if f.endswith(".yaml") or f.endswith(".yml")]
        if not yaml_files:
            raise FileNotFoundError(f"No yaml config found in {path}")
        config_path = _os.path.join(path, yaml_files[0])
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

        init_params = config["Model"]["__init_params__"]

        tokenizer_path = _os.path.join(path, "tokenizer.json")
        if _os.path.exists(tokenizer_path):
            init_params["tokenizer_path"] = tokenizer_path

        model = cls(**init_params)

        model._load_weights_from_registry(path)

        return model

    def _load_weights_from_registry(self, weight_dir):
        import os as _os, json as _json

        ckpt_dir = _os.path.join(weight_dir, "checkpoints")
        if not _os.path.isdir(ckpt_dir):
            ckpt_dir = weight_dir

        index_path = _os.path.join(ckpt_dir, "model.pdparams.index.json")
        if _os.path.exists(index_path):
            self._load_sharded_weights(ckpt_dir, index_path)
            return

        best_path = _os.path.join(ckpt_dir, "best.pdparams")
        if _os.path.exists(best_path):
            self._load_single_weight(best_path)
            return

        raise FileNotFoundError(
            f"No weights found in {ckpt_dir}: "
            f"expected best.pdparams or model.pdparams.index.json"
        )

    def _load_sharded_weights(self, ckpt_dir, index_path):
        import json as _json, os as _os

        with open(index_path) as f:
            idx = _json.load(f)

        model_sd = self.state_dict()
        shard_files = sorted(set(idx["weight_map"].values()))
        for shard_file in shard_files:
            shard_path = _os.path.join(ckpt_dir, shard_file)
            shard = paddle.load(shard_path, return_numpy=True)
            for key, val in shard.items():
                if key in model_sd:
                    w = paddle.to_tensor(val)
                    target = model_sd[key]
                    if w.dtype != target.dtype:
                        w = w.cast(target.dtype)
                    target.set_value(w)
            del shard

    def _load_single_weight(self, weight_path):
        state_dict = paddle.load(weight_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.set_state_dict(state_dict)

    def _run_qformer(self, query_tokens, material_embeds, material_atts, prompt_text=None):
        """Run Q-Former with optional text input, shared by forward/generate/generate_followup.

        Args:
            query_tokens: [B, num_query_tokens, hidden_size]
            material_embeds: [B, N_atoms, encoder_width]
            material_atts: [B, N_atoms]
            prompt_text: list of strings or None

        Returns:
            Q-Former output (BaseModelOutputWithPoolingAndCrossAttentions)
        """
        if self.qformer_text_input and prompt_text is not None:
            text_input = self.tokenizer(
                prompt_text,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pd",
            )
            query_atts = paddle.ones(query_tokens.shape[:-1], dtype=paddle.int64)
            qformer_atts = paddle.concat([query_atts, text_input.attention_mask], axis=1)
            return self.Qformer.bert(
                text_input.input_ids,
                attention_mask=qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=material_embeds,
                encoder_attention_mask=material_atts,
                return_dict=True,
            )
        else:
            return self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=material_embeds,
                encoder_attention_mask=material_atts,
                return_dict=True,
            )

    def _encode_for_generate(self, samples):
        """Shared encoding pipeline for generate / generate_followup.

        Steps: CHGNet encode -> Q-Former encode -> llm_proj -> tokenizer -> concat.
        Also sets llm_tokenizer.padding_side to "left".

        Returns:
            (inputs_embeds, attention_mask, prompt_list)
        """
        self.llm_tokenizer.padding_side = "left"
        prompt = samples.get("prompt", self.prompt)
        if isinstance(prompt, str):
            prompt = [prompt]

        # Material encoding
        material_embed = self.material_encoder.predict_structure_embedding(
            samples["material_sample"]
        ).unsqueeze(0)
        material_att = paddle.ones(material_embed.shape[:-1], dtype=paddle.int64)

        # Q-Former encoding
        query_tokens = self.query_tokens.expand([1, -1, -1])
        query_output = self._run_qformer(query_tokens, material_embed, material_att, prompt)

        # llm_proj mapping
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.shape[1], :])
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        # Tokenizer encoding + concatenation
        prompt_tokens = self.llm_tokenizer(prompt, return_tensors="pd", padding="longest")
        input_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        inputs_embeds = paddle.concat([inputs_llm, input_embeds], axis=1)
        attention_mask = paddle.concat([atts_llm, prompt_tokens.attention_mask], axis=1)
        return inputs_embeds, attention_mask, prompt

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        input_ids_list, attention_mask_list = [], []

        for i in range(input_ids.shape[0]):
            this_input_len = input_atts[i].sum()
            input_part_targets_len.append(this_input_len)

            input_ids_list.append(
                paddle.concat(
                    [
                        input_ids[i][:this_input_len],
                        output_ids[i][1:],
                        input_ids[i][this_input_len:],
                    ]
                )
            )

            attention_mask_list.append(
                paddle.concat(
                    [
                        input_atts[i][:this_input_len],
                        output_atts[i][1:],
                        input_atts[i][this_input_len:],
                    ]
                )
            )

        return {
            "input_ids": paddle.stack(input_ids_list),
            "attention_mask": paddle.stack(attention_mask_list),
        }, input_part_targets_len

    def forward(self, samples, embedding_list, embedding_mask):
        material_embeds = paddle.stack(embedding_list)
        material_atts = paddle.stack(embedding_mask)
        query_tokens = self.query_tokens.expand([material_embeds.shape[0], -1, -1])

        # Shared Q-Former encoding (handles qformer_text_input branching)
        text_list = samples.get("text_input", None)
        query_output = self._run_qformer(query_tokens, material_embeds, material_atts, text_list)

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:, : query_tokens.shape[1], :])
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = "left"

        if "text_input" in samples:
            text_input_tokens = self.llm_tokenizer(
                samples["text_input"],
                return_tensors="pd",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            )
        else:
            bos_token = self.llm_tokenizer.bos_token
            input_text = [bos_token] * material_embeds.shape[0]
            text_input_tokens = self.llm_tokenizer(
                input_text,
                return_tensors="pd",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            )

        self.llm_tokenizer.truncation_side = "right"
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text"]],
            return_tensors="pd",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        )

        llm_tokens, input_target_lengths = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100
        )
        for i, length in enumerate(input_target_lengths):
            targets[i][:length] = -100

        empty_targets = paddle.ones(atts_llm.shape, dtype=paddle.int64).fill_(-100)
        targets = paddle.concat([empty_targets, targets], axis=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        inputs_embeds = paddle.concat(
            [inputs_llm.cast(inputs_embeds.dtype), inputs_embeds], axis=1
        )
        attention_mask = paddle.concat([atts_llm, llm_tokens["attention_mask"]], axis=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        return {"loss": outputs.loss}

    @paddle.no_grad()
    def chat(self, structure, prompt, max_new_tokens=64):
        """Single-turn dialogue: given a crystal structure and a text prompt,
        return the generated answer (greedy decoding, fp32).

        This is a convenience wrapper around the CHGNet -> Q-Former ->
        llm_proj -> Mistral pipeline for inference / dialogue use.

        Args:
            structure: pymatgen ``Structure`` object.
            prompt: text question, e.g. "what is the chemical formula?".
            max_new_tokens: max number of new tokens to generate.

        Returns:
            str: decoded answer text.
        """
        embed = self.material_encoder.predict_structure_embedding(structure).unsqueeze(0)
        mask = paddle.ones(embed.shape[:-1], dtype=paddle.int64)
        qt = self.query_tokens.expand([1, -1, -1])
        qo = self._run_qformer(qt, embed, mask, [prompt])
        inputs_llm = self.llm_proj(qo.last_hidden_state[:, : qt.shape[1], :])
        atts_llm = paddle.ones(inputs_llm.shape[:-1], dtype=paddle.int64)
        self.llm_tokenizer.padding_side = "left"
        pt = self.llm_tokenizer([prompt], return_tensors="pd", padding="longest")
        ie = self.llm_model.get_input_embeddings()(pt.input_ids)
        ie = paddle.concat([inputs_llm, ie], axis=1)
        am = paddle.concat([atts_llm, pt.attention_mask], axis=1)
        out = self.llm_model.generate(
            inputs_embeds=ie,
            attention_mask=am,
            do_sample=False,
            max_length=ie.shape[1] + max_new_tokens,
        )
        out[out == 0] = 2
        return self.llm_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @paddle.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        inputs_embeds, attention_mask, _prompt = self._encode_for_generate(samples)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # sanitize output
        return [self.llm_tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

    def generate_followup(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        inputs_embeds, attention_mask, prompt = self._encode_for_generate(samples)

        with self.maybe_autocast():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2  # sanitize output
        decoded_outputs = [
            self.llm_tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs
        ]

        cleaned_outputs = []
        for out, p in zip(decoded_outputs, prompt * num_captions):
            if out.lower().startswith(p.lower()):
                out = out[len(p) :].lstrip(":,.- \n")
            cleaned_outputs.append(out)

        return cleaned_outputs
