# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import inspect
import torch
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoModelForCausalLM
from transformers.generation import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.utils import logging

from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceTokenizerEncoderOutput
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice import VibeVoiceConfig
from .modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer, VibeVoiceTextTokenizerFast
from .modeling_vibevoice import VibeVoiceModel, VibeVoicePreTrainedModel
from .streamer import AudioStreamer, AsyncAudioStreamer

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]


# ============================================================================
# Transformers >= 4.57 Compatibility Layer
# (stesse tecniche di file 2: cache refactor / DynamicCache)
# ============================================================================

class MockCacheLayer:
    """
    Mock cache layer per compatibilità transformers >= 4.57.
    Espone l'interfaccia `layers` attesa da DynamicCache in alcune code path.
    """

    def __init__(self, key_cache, value_cache, parent_cache=None, layer_idx=0):
        self.key_cache = key_cache
        self.value_cache = value_cache
        self._parent_cache = parent_cache
        self._layer_idx = layer_idx

    def get_mask_sizes(self, cache_position):
        kv_length = self.key_cache.shape[2] if self.key_cache is not None else 0
        return kv_length, 0

    def update(self, key_states, value_states, cache_kwargs=None):
        if self._parent_cache is None:
            return self.key_cache, self.value_cache

        parent = self._parent_cache
        idx = self._layer_idx

        while len(parent.key_cache) <= idx:
            parent.key_cache.append(None)
            parent.value_cache.append(None)

        if parent.key_cache[idx] is not None:
            parent.key_cache[idx] = torch.cat([parent.key_cache[idx], key_states], dim=2)
            parent.value_cache[idx] = torch.cat([parent.value_cache[idx], value_states], dim=2)
        else:
            parent.key_cache[idx] = key_states
            parent.value_cache[idx] = value_states

        self.key_cache = parent.key_cache[idx]
        self.value_cache = parent.value_cache[idx]
        return self.key_cache, self.value_cache


def _ensure_cache_has_layers(cache):
    """
    Garantisce che `cache` abbia gli attributi attesi in transformers recenti:
    - layer_class_to_replicate, offloading, is_compileable
    - layers: lista di oggetti layer-like con key_cache/value_cache e update()
    """
    if cache is None:
        return cache

    for attr, default in [("layer_class_to_replicate", None), ("offloading", False), ("is_compileable", False)]:
        if not hasattr(cache, attr):
            try:
                setattr(cache, attr, default)
            except AttributeError:
                pass

    # Se esistono key_cache/value_cache (lista per layer) costruiamo layers
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            cache.layers = [
                MockCacheLayer(cache.key_cache[i], cache.value_cache[i], parent_cache=cache, layer_idx=i)
                for i in range(len(cache.key_cache))
            ]
        except Exception:
            # non bloccare mai se è read-only o implementazione diversa
            pass
    elif not hasattr(cache, "layers"):
        try:
            cache.layers = []
        except Exception:
            pass

    return cache


def _update_model_kwargs_for_generation_windowed(
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    num_new_tokens: int = 1,
) -> Dict[str, Any]:
    """
    Variante “window-aware” come in file 2:
    aggiorna past_key_values, attention_mask e cache_position in modo consistente
    quando aggiungi >1 token per step (o quando il chiamante vuole preallocare finestre).
    """
    model_kwargs["past_key_values"] = _ensure_cache_has_layers(outputs.past_key_values)

    attention_mask = model_kwargs["attention_mask"]
    model_kwargs["attention_mask"] = torch.cat(
        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
    )

    cache_pos = model_kwargs["cache_position"]
    model_kwargs["cache_position"] = torch.arange(
        cache_pos[-1] + 1, cache_pos[-1] + num_new_tokens + 1, device=cache_pos.device
    )

    return model_kwargs


# ============================================================================
# Outputs / Logits processor
# ============================================================================

@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """
    Output type for VibeVoice generation.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences.
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None


class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    """Constrains token generation to only valid tokens during speech generation."""

    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.valid_token_ids] = 0
        return scores + mask


# ============================================================================
# Main model
# ============================================================================

class VibeVoiceForConditionalGenerationInference(VibeVoicePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)

        # Initialize the base model
        self.model = VibeVoiceModel(config)

        # LM head for text generation
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)

        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        if not getattr(self.config, "tie_word_embeddings", False):
            return

        if hasattr(self, "lm_head") and hasattr(self.model.language_model, "embed_tokens"):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        """Process speech inputs through tokenizers and connectors."""
        with torch.no_grad():
            if speech_type == "audio":
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]

                acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * \
                                    self.model.speech_scaling_factor.to(acoustic_latents.device)

                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]
                return acoustic_features, acoustic_connected

            elif speech_type == "pt":
                encoder_output = VibeVoiceTokenizerEncoderOutput(mean=speech_tensors, std=self.acoustic_tokenizer.config.fix_std)
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]

                acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * \
                                    self.model.speech_scaling_factor.to(acoustic_latents.device)

                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks.cpu()]
                return acoustic_features, acoustic_connected

            else:
                raise NotImplementedError(f"Speech type {speech_type} not implemented")

    # ---------------------------------------------------------------------
    # Transformers >= 4.57: prepare_inputs_for_generation compat (da file 2)
    # ---------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        """
        Prepare model inputs for generation (transformers >= 4.57 friendly).

        - Gestisce slicing coerente quando past_key_values è presente.
        - Supporta input_ids vs inputs_embeds in modo compatibile con cache_position.
        - Costruisce/slice position_ids da attention_mask (se non forniti).
        """
        model_inputs = {"cache_position": cache_position}

        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

            if inputs_embeds is not None and input_ids.shape[1] == 0:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0]:]
            elif inputs_embeds is not None or (cache_position is not None and cache_position[-1] >= input_ids.shape[1]):
                input_ids = input_ids[:, -cache_position.shape[0]:]
            elif cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        use_embeds = inputs_embeds is not None and (
            past_key_values is None or (cache_position is not None and len(cache_position) == inputs_embeds.shape[1])
        )
        if use_embeds:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids.clone(memory_format=torch.contiguous_format) if input_ids is not None else None
            model_inputs["inputs_embeds"] = None

        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # Create position_ids from attention_mask if not passed
        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids

        # Slice position_ids when using cache
        if kwargs.get("position_ids") is not None:
            if past_key_values is not None:
                seq_len = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs["input_ids"].shape[1]
                )
                model_inputs["position_ids"] = kwargs["position_ids"][:, -seq_len:].clone(memory_format=torch.contiguous_format)
            else:
                model_inputs["position_ids"] = kwargs.pop("position_ids").clone(memory_format=torch.contiguous_format)

        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        model_inputs.pop("labels", None)
        return model_inputs

    def _init_cache_for_generation(self, generation_config, model_kwargs, batch_size, max_cache_length, device):
        """
        Cache init robusta come in file 2:
        - transformers >= 4.57: restituisce None (lascia creare al modello/DynamicCache)
        - transformers < 4.57: usa _prepare_cache_for_generation (con o senza parametro device)
        """
        try:
            from transformers.cache_utils import DynamicCache
            sig = inspect.signature(DynamicCache.__init__)
            if "config" in sig.parameters:
                # transformers >= 4.57: il modello gestisce cache dinamicamente
                return None
            else:
                # più vecchie: fallback a _prepare_cache_for_generation
                prep_sig = inspect.signature(self._prepare_cache_for_generation)
                if "device" in prep_sig.parameters:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
                else:
                    self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length)
                return model_kwargs.get("past_key_values")
        except Exception:
            return None

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        num_new_tokens=1,
    ):
        """
        Override: dopo il super(), normalizza past_key_values per compatibilità >= 4.57.
        """
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, num_new_tokens=num_new_tokens
        )
        if "past_key_values" in model_kwargs:
            model_kwargs["past_key_values"] = _ensure_cache_has_layers(model_kwargs["past_key_values"])
        return model_kwargs

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if speech_tensors is not None and speech_masks is not None:
            _, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    # ---------------------------------------------------------------------
    # Generation config builder (aggiornata: init cache robusta)
    # ---------------------------------------------------------------------
    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config,
            True,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            **kwargs,
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = self.device

        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1

        # >>> compat: init cache robusto (>=4.57 -> None; <4.57 -> prepara)
        model_kwargs["past_key_values"] = self._init_cache_for_generation(
            generation_config, model_kwargs, batch_size, max_cache_length, device
        )

        model_kwargs["cache_position"] = torch.arange(input_ids_length, device=device, dtype=torch.long)

        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)

        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )
            stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=StoppingCriteriaList()
            )
            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria

        return generation_config, model_kwargs, input_ids

    # ---------------------------------------------------------------------
    # Generate (logica originale + compat cache)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        is_prefill: bool = True,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        stop_check_fn: Optional[Callable[[], bool]] = None,
        tqdm_class: Optional[type] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        tokenizer = kwargs.pop("tokenizer", None)
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        max_length_times = kwargs.pop("max_length_times", 2)

        if kwargs.get("max_new_tokens", None) is None:
            kwargs["max_new_tokens"] = self.config.decoder_config.max_position_embeddings - kwargs["input_ids"].shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )

        negative_kwargs = {
            "input_ids": torch.full(
                (kwargs["input_ids"].shape[0], 1),
                tokenizer.speech_start_id,
                dtype=torch.long,
                device=kwargs["input_ids"].device,
            ),
            "attention_mask": torch.ones(
                (kwargs["input_ids"].shape[0], 1),
                dtype=torch.long,
                device=kwargs["input_ids"].device,
            ),
            "max_new_tokens": kwargs.get("max_new_tokens", 100),
        }
        _, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()

        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        inputs_embeds = None
        verbose = kwargs.get("verbose", False)

        audio_chunks = [[] for _ in range(batch_size)]

        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs["attention_mask"].sum(dim=-1)

        valid_tokens = [
            generation_config.speech_start_id,
            generation_config.speech_end_id,
            generation_config.speech_diffusion_id,
            generation_config.eos_token_id,
        ]
        if getattr(generation_config, "bos_token_id", None) is not None:
            valid_tokens.append(generation_config.bos_token_id)

        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(token_constraint_processor)

        max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
        max_step_per_sample = torch.min(
            generation_config.max_length - initial_length_per_sample,
            (max_length_times * initial_length_per_sample).long(),
        )
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if kwargs.get("show_progress_bar", True):
            tqdm_fn = tqdm_class if tqdm_class is not None else tqdm
            progress_bar = tqdm_fn(range(max_steps), desc="Generating", leave=False)
        else:
            progress_bar = range(max_steps)

        for step in progress_bar:
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                if audio_streamer is not None:
                    audio_streamer.end()
                break

            if audio_streamer is not None and hasattr(audio_streamer, "finished_flags"):
                if any(audio_streamer.finished_flags):
                    if verbose:
                        print(f"Audio generation stopped externally at step {step + 1}")
                    break

            if finished_tags.all():
                if hasattr(progress_bar, "set_description"):
                    progress_bar.set_description("Generation complete")
                break

            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break

            if hasattr(progress_bar, "set_description"):
                active_samples = (~finished_tags).sum().item()
                progress_bar.set_description(f"Generating (active: {active_samples}/{batch_size})")

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if is_prefill:
                prefill_inputs = {}
                if speech_tensors is not None:
                    prefill_inputs["speech_tensors"] = speech_tensors.to(device=device)
                if speech_masks is not None:
                    prefill_inputs["speech_masks"] = speech_masks.to(device)
                if speech_input_mask is not None:
                    prefill_inputs["speech_input_mask"] = speech_input_mask.to(device)
                is_prefill = False
            else:
                _ = model_inputs.pop("inputs_embeds", None)
                prefill_inputs = {"inputs_embeds": inputs_embeds}

            outputs = self(
                **model_inputs,
                **prefill_inputs,
                logits_to_keep=1,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_tokens[finished_tags] = generation_config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # Negative pass logic (originale)
            if not kwargs.get("refresh_negative", True):
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                if negative_model_inputs.get("inputs_embeds") is None and inputs_embeds is not None:
                    negative_model_inputs["inputs_embeds"] = inputs_embeds
                    negative_model_inputs["input_ids"] = None

                negative_outputs = self(
                    **negative_model_inputs,
                    logits_to_keep=0,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

            # EOS
            if (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    finished_tags[new_eos_indices] = True
                    if verbose:
                        print(f"Samples {new_eos_indices.tolist()} reached EOS token at step {step + 1}.", flush=True)
                    if audio_streamer is not None:
                        audio_streamer.end(new_eos_indices)

            # per-sample max length
            max_length_reached = step >= max_step_per_sample
            new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
            if new_max_length_indices.numel() > 0:
                finished_tags[new_max_length_indices] = True
                reach_max_step_sample[new_max_length_indices] = True
                if verbose:
                    print(
                        f"Samples {new_max_length_indices.tolist()} reached max generation length at step {step + 1}.",
                        flush=True,
                    )
                if audio_streamer is not None:
                    audio_streamer.end(new_max_length_indices)

            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)

            # speech_begin: refresh negative (originale) — ora protetto da ensure_cache layers a monte
            diffusion_start_indices = torch.arange(batch_size, device=device)[
                ~finished_tags & (next_tokens == generation_config.speech_start_id)
            ]
            if diffusion_start_indices.numel() > 0 and kwargs.get("refresh_negative", True):
                for sample_idx in diffusion_start_indices.tolist():
                    negative_model_kwargs["attention_mask"][sample_idx, :] = 0
                    negative_model_kwargs["attention_mask"][sample_idx, -1] = 1

                pkv = negative_model_kwargs.get("past_key_values", None)
                if pkv is not None and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                    for layer_idx, (k_cache, v_cache) in enumerate(zip(pkv.key_cache, pkv.value_cache)):
                        for sample_idx in diffusion_start_indices.tolist():
                            k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                            v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()

                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id

            # embeddings per prossimo step
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)

            diffusion_indices = torch.arange(batch_size, device=device)[
                ~finished_tags & (next_tokens == generation_config.speech_diffusion_id)
            ]

            if diffusion_indices.numel() > 0:
                if kwargs.get("refresh_negative", True):
                    negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                    if negative_model_inputs.get("inputs_embeds") is None and inputs_embeds is not None:
                        negative_model_inputs["inputs_embeds"] = inputs_embeds
                        negative_model_inputs["input_ids"] = None

                    negative_outputs = self(
                        **negative_model_inputs,
                        logits_to_keep=0,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    negative_model_kwargs = self._update_model_kwargs_for_generation(
                        negative_outputs, negative_model_kwargs, is_encoder_decoder=False
                    )
                    negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                    start_indices = correct_cnt[non_diffusion_indices]

                    seq_len = negative_model_kwargs["attention_mask"].shape[1]
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs["attention_mask"][sample_idx, start_idx + 1 :] = \
                                negative_model_kwargs["attention_mask"][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs["attention_mask"][sample_idx, start_idx] = 0

                    pkv = negative_model_kwargs.get("past_key_values", None)
                    if pkv is not None and hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
                        for k_cache, v_cache in zip(pkv.key_cache, pkv.value_cache):
                            for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                                if start_idx + 1 < k_cache.shape[2] - 1:
                                    k_cache[sample_idx, :, start_idx + 1 :, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                    v_cache[sample_idx, :, start_idx + 1 :, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()

                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx + 1 :] = negative_input_ids[sample_idx, start_idx:-1].clone()

                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]

                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)

                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - \
                                self.model.speech_bias_factor.to(speech_latent.device)

                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False,
                )

                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                if audio_streamer is not None:
                    audio_streamer.put(audio_chunk, diffusion_indices)

                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False,
                ).mean

                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed
                next_inputs_embeds[diffusion_indices] = diffusion_embeds

            inputs_embeds = next_inputs_embeds

        if audio_streamer is not None:
            audio_streamer.end()

        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                final_audio_outputs.append(torch.cat(sample_chunks, dim=-1))
            else:
                final_audio_outputs.append(None)

        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(self.model.prediction_head.device)
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]


AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGenerationInference)

__all__ = [
    "VibeVoiceForConditionalGenerationInference",
]
