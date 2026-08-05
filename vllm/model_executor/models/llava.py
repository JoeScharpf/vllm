# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Final, Literal, Protocol, TypeAlias, TypeVar

import torch
import torch.nn as nn
from transformers import (
    BatchFeature,
    CLIPVisionConfig,
    LlavaConfig,
    PixtralVisionConfig,
    PretrainedConfig,
    SiglipVisionConfig,
)
from transformers.models.llava import LlavaProcessor
from transformers.models.pixtral import PixtralProcessor

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
from vllm.multimodal.dart_scoring import dart_prefix_states_llama
from vllm.multimodal.hiprune import (
    HIPRUNE_CONFIG_WIDTH,
    LLAVA_OBJECT_LAYER,
    anchorprune_resolve_kmin,
    anchorprune_select,
    build_anchorprune_metadata,
    build_dart_metadata,
    build_checkered_metadata,
    build_hiprune_metadata,
    build_hiprune_pp_metadata,
    build_hydart_metadata,
    build_nprune_metadata,
    checkered_keep_count,
    checkered_select,
    dart_keep_count,
    dart_select,
    get_dart_layer,
    get_dart_pivots,
    get_hiprune_method,
    get_hiprune_pp_beta,
    get_hiprune_prompt,
    get_hiprune_ratio,
    get_nprune_stride,
    hiprune_pp_select,
    hiprune_select,
    hydart_select,
    nprune_keep_count,
    nprune_select,
    pack_hiprune_config,
    unpack_hiprune_config,
    HIPRUNE_MM_KWARG_KEYS,
    HipruneConfig,
)
from vllm.multimodal.hiprune import (
    compute_hiprune_pp_budget as hiprune_pp_budget,
)
from vllm.multimodal.hiprune import (
    compute_retained_tokens_count as hiprune_retained_tokens_count,
)
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .clip import CLIPVisionModel
from .interfaces import (
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .module_mapping import MultiModelKeys
from .pixtral import PixtralHFEncoderInfo, PixtralHFVisionModel
from .siglip import SiglipVisionModel
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_layer_index,
    init_vllm_registered_model,
    maybe_prefix,
)
from .vision import get_num_selected_vision_tokens, get_vision_encoder_info


class LlavaImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels (3)
        - h: Height
        - w: Width

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    type: Literal["pixel_values"] = "pixel_values"
    pixel_values: Annotated[torch.Tensor, TensorShape("bn", 3, "h", "w")]

    # HiPrune retention ratio per image (fraction of patch tokens kept),
    # emitted by the processor only when pruning is requested.
    hiprune_ratio: Annotated[
        torch.Tensor | None,
        TensorShape("bn"),
    ] = None

    # HiPrune++ / DART prompt token ids, one identical row per image (the
    # mm field machinery is per-item), emitted by the processor whenever
    # pruning is requested (an empty row for methods that don't use the
    # prompt, keeping field presence uniform across a mixed-method
    # batch). "pl" (prompt length) is dynamic because a batch can span
    # requests with different prompts (rows then arrive as a list of
    # unequal-length tensors).
    hiprune_prompt_ids: Annotated[
        torch.Tensor | list[torch.Tensor] | None,
        TensorShape("bn", "pl", dynamic_dims={"pl"}),
    ] = None

    # Packed per-image pruning config (method id + knobs; see
    # pack_hiprune_config), emitted alongside hiprune_ratio so the model
    # forward can dispatch the selection method per image — a batch may
    # span requests with different methods.
    hiprune_config: Annotated[
        torch.Tensor | None,
        TensorShape("bn", HIPRUNE_CONFIG_WIDTH),
    ] = None


class PixtralHFImagePixelInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - c: Number of channels
        - h: Height
        - w: Width

    Note that `height` or `width` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    type: Literal["pixel_values_pixtral"] = "pixel_values_pixtral"
    pixel_values: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "c", "h", "w", dynamic_dims={"h", "w"}),
    ]


class LlavaImageEmbeddingInputs(TensorSchema):
    """
    Dimensions:
        - bn: Batch size * number of images
        - ifs: Image feature size
        - hs: Hidden size (must match language model backbone)
    """

    type: Literal["image_embeds"] = "image_embeds"
    data: Annotated[torch.Tensor, TensorShape("bn", "ifs", "hs")]


LlavaImageInputs: TypeAlias = (
    LlavaImagePixelInputs | PixtralHFImagePixelInputs | LlavaImageEmbeddingInputs
)
"""Alias for supported LLaVA image input types."""


class LlavaMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        multimodal_projector_bias: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.act = get_act_fn(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            text_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


class LlavaLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_index: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[int | list[int]]


class LlavaLikeProcessor(Protocol):
    image_token: Final[str]


class BaseLlavaProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> LlavaLikeConfig:
        return self.ctx.get_hf_config(LlavaConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> LlavaLikeProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        return get_num_selected_vision_tokens(
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
            hf_config.vision_feature_select_strategy,
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        vision_encoder_info = self.get_vision_encoder_info()
        width = height = vision_encoder_info.get_image_size()
        return ImageSize(width=width, height=height)

    def get_max_image_tokens(self) -> int:
        target_width, target_height = self.get_image_size_with_most_features()

        return self.get_num_image_tokens(
            image_width=target_width,
            image_height=target_height,
        )


_I = TypeVar("_I", bound=BaseLlavaProcessingInfo)


class LlavaDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = self.info.get_image_size_with_most_features()

        image_overrides = mm_options.get("image")

        return {
            "image": self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class LlavaProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        # hiprune_* are vLLM-side kwargs (placeholder sizing + model
        # selection); the HF processor does not know them.
        for key in HIPRUNE_MM_KWARG_KEYS:
            kwargs.pop(key, None)
        hf_processor = self.ctx.get_hf_processor(LlavaProcessor, **kwargs)
        # In case patch_size is omitted from `processor_config.json`
        # e.g. for E5-V: https://huggingface.co/royokong/e5-v
        if hf_processor.patch_size is None:
            patch_size = self.get_vision_encoder_info().get_patch_size()
            hf_processor.patch_size = patch_size
        return hf_processor


class BaseLlavaMultiModalProcessor(BaseMultiModalProcessor[_I]):
    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


class LlavaMultiModalProcessor(BaseLlavaMultiModalProcessor[LlavaProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            hiprune_ratio=MultiModalFieldConfig.batched("image"),
            hiprune_prompt_ids=MultiModalFieldConfig.batched("image"),
            hiprune_config=MultiModalFieldConfig.batched("image"),
        )

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        hiprune_ratio = get_hiprune_ratio(mm_kwargs)
        hiprune_prompt = get_hiprune_prompt(mm_kwargs)
        if hiprune_ratio is not None:
            mm_config = self.info.ctx.get_mm_config()
            if not mm_config.is_multimodal_pruning_enabled():
                raise ValueError(
                    "hiprune_ratio (token_pruning) requires the server to "
                    "be started with --enable-hiprune."
                )
        # The HF processor does not know the vLLM-side kwargs.
        hiprune_kwargs = mm_kwargs
        mm_kwargs = {
            k: v for k, v in mm_kwargs.items() if k not in HIPRUNE_MM_KWARG_KEYS
        }

        processed_outputs = super()._call_hf_processor(
            prompt, mm_data, mm_kwargs, tok_kwargs
        )

        # HiPrune: attach the retention ratio and packed method/knob
        # config per image so they reach the model at encode time (they
        # are per-request). The prompt placeholder count is reduced in
        # _get_prompt_updates with the same count function the model
        # uses at selection time.
        if hiprune_ratio is not None and "pixel_values" in processed_outputs:
            num_images = len(processed_outputs["pixel_values"])
            processed_outputs["hiprune_ratio"] = torch.full(
                (num_images,), hiprune_ratio, dtype=torch.float32
            )
            processed_outputs["hiprune_config"] = (
                pack_hiprune_config(hiprune_kwargs)
                .unsqueeze(0)
                .repeat(num_images, 1)
            )
            # HiPrune++ / DART: ship the prompt token ids to the model so
            # it can embed them (LM embedding table) for text-guided
            # selection (HiPrune++) or text pivots (DART). Always
            # attached when pruning is requested — an empty row for
            # methods that don't use the prompt — so field presence is
            # uniform when a batch mixes methods across requests. One
            # identical row per image — the mm field machinery is
            # per-item.
            ids: list[int] = []
            if get_hiprune_method(hiprune_kwargs) in (
                "hiprune_pp",
                "dart",
                "anchorprune",
            ):
                tokenizer = self.info.get_tokenizer()
                if hiprune_prompt:
                    ids = tokenizer.encode(hiprune_prompt, add_special_tokens=False)
            processed_outputs["hiprune_prompt_ids"] = torch.tensor(
                [ids], dtype=torch.long
            ).repeat(num_images, 1)

        return processed_outputs

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index
        hiprune_ratio = get_hiprune_ratio(hf_processor_mm_kwargs)

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )
                # HiPrune: shrink the image placeholder run to the
                # retained budget. The model keeps exactly this many
                # tokens at encode time (same count function on both
                # sides). HiPrune++ adds the text-guided tokens on top
                # of the base budget (additive, per the paper); the
                # count never depends on the prompt content.
                if hiprune_ratio is not None:
                    method = get_hiprune_method(hf_processor_mm_kwargs)
                    if method == "hiprune_pp":
                        base, t_sum = hiprune_pp_budget(
                            num_image_tokens,
                            hiprune_ratio,
                            get_hiprune_pp_beta(hf_processor_mm_kwargs),
                        )
                        num_image_tokens = base + t_sum
                    elif method == "dart":
                        p_img, p_txt = get_dart_pivots(hf_processor_mm_kwargs)
                        num_image_tokens = dart_keep_count(
                            num_image_tokens, hiprune_ratio, p_img, p_txt
                        )
                    elif method == "nprune":
                        # Grid-shape-aware count (exact lattice). LLaVA's
                        # patch grid is square (24x24 for 336px CLIP).
                        side = math.isqrt(num_image_tokens)
                        assert side * side == num_image_tokens
                        num_image_tokens = nprune_keep_count(
                            side,
                            side,
                            get_nprune_stride(hf_processor_mm_kwargs),
                        )
                    elif method == "checkered":
                        # Exact ceil(N/2), shape-independent — NOT the
                        # ratio path (round(n*0.5) rounds half-to-even).
                        num_image_tokens = checkered_keep_count(
                            num_image_tokens
                        )
                    else:
                        num_image_tokens = hiprune_retained_tokens_count(
                            num_image_tokens, hiprune_ratio
                        )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


class PixtralHFProcessingInfo(BaseLlavaProcessingInfo):
    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(PixtralProcessor, **kwargs)


class PixtralHFMultiModalProcessor(BaseMultiModalProcessor[PixtralHFProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )

        pixel_values = processed_outputs.get("pixel_values")
        if pixel_values is not None:
            # Avoid padding since we need the output for each image to be
            # independent of other images for the cache to work correctly
            image_sizes = processed_outputs["image_sizes"]
            assert len(pixel_values) == len(image_sizes)

            processed_outputs["pixel_values"] = [
                p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)
            ]

        return processed_outputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_break_id = vocab[processor.image_break_token]
        image_token_id = hf_config.image_token_index
        image_end_id = vocab[processor.image_end_token]

        assert isinstance(hf_config.vision_config, PixtralVisionConfig)
        encoder_info = PixtralHFEncoderInfo(hf_config)

        def get_replacement(item_idx: int):
            images = mm_items.get_items("image", ImageProcessorItems)
            image_size = images.get_image_size(item_idx)

            ncols, nrows = encoder_info.get_patch_grid_size(
                image_width=image_size.width,
                image_height=image_size.height,
            )

            tokens = ([image_token_id] * ncols + [image_break_id]) * nrows
            tokens[-1] = image_end_id

            return PromptUpdateDetails.select_token_id(tokens, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


def _build_llava_or_pixtral_hf_info(
    ctx: InputProcessingContext,
) -> BaseLlavaProcessingInfo:
    hf_config = ctx.get_hf_config(LlavaConfig)

    if isinstance(hf_config.vision_config, PixtralVisionConfig):
        return PixtralHFProcessingInfo(ctx)

    return LlavaProcessingInfo(ctx)


def _build_llava_or_pixtral_hf_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, PixtralHFProcessingInfo):
        return PixtralHFMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    if isinstance(info, LlavaProcessingInfo):
        return LlavaMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    raise NotImplementedError(type(info))


def _get_num_hidden_layers(hf_config: LlavaLikeConfig) -> int:
    """Determine the number of hidden layers to initialize up to in the
    visual encoder.

    Args:
        hf_config: Model config with vision feature layer(s).
    """
    feature_layers = hf_config.vision_feature_layer
    num_hidden_layers = hf_config.vision_config.num_hidden_layers
    # If we have one feature layer, initialize up to that layer
    if isinstance(feature_layers, int):
        return get_layer_index(feature_layers, num_hidden_layers)
    # If we have multiple feature layers, initialize up to the deepest one
    elif isinstance(feature_layers, (list, tuple)):
        return max(get_layer_index(idx, num_hidden_layers) for idx in feature_layers)
    raise TypeError(
        f"vision_layer_feature type: {type(feature_layers)} is not supported"
    )


def init_vision_tower_for_llava(
    hf_config: LlavaLikeConfig,
    quant_config: QuantizationConfig | None,
    *,
    require_post_norm: bool | None = None,
    prefix: str = "",
) -> CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel:
    vision_config = hf_config.vision_config

    # Initialize the vision tower only up to the deepest required feature layer
    num_hidden_layers = _get_num_hidden_layers(hf_config)

    if isinstance(vision_config, CLIPVisionConfig):
        return CLIPVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, SiglipVisionConfig):
        return SiglipVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )
    elif isinstance(vision_config, PixtralVisionConfig):
        return PixtralHFVisionModel(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers,
            require_post_norm=require_post_norm,
            prefix=prefix,
        )

    msg = f"Unsupported vision config: {type(vision_config)}"
    raise NotImplementedError(msg)


@MULTIMODAL_REGISTRY.register_processor(
    _build_llava_or_pixtral_hf_processor,
    info=_build_llava_or_pixtral_hf_info,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class LlavaForConditionalGeneration(
    nn.Module,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.configure_mm_token_handling(
            vocab_size=config.text_config.vocab_size,
            mm_token_ids=[config.image_token_index],
        )

        # NOTE: These are special cases for Pixtral-12B in the HF-format
        # https://huggingface.co/mistral-community/pixtral-12b/blob/main/config.json  # noqa
        if (
            config.text_config.architectures is None
            and config.text_config.model_type == "mistral"
        ):
            config.text_config.architectures = ["MistralForCausalLM"]
        if (
            config.projector_hidden_act is None
            and config.vision_config.hidden_act == "gelu"
        ):
            config.projector_hidden_act = "gelu"

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = init_vision_tower_for_llava(
                config,
                quant_config=quant_config,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LlavaMultiModalProjector(
                vision_hidden_size=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                projector_hidden_act=config.projector_hidden_act,
                multimodal_projector_bias=config.multimodal_projector_bias,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # HiPrune state for the most recent embed_multimodal call: per-item
        # pruning metadata, read by the model runner and returned via
        # token_pruning_metadata. LLaVA needs no kept-mask handoff (no
        # mrope): shrinking the placeholder run already yields contiguous
        # positions for the kept tokens.
        self._hiprune_metadata: list[dict[str, object] | None] = []
        self.hiprune_metadata_per_item: list[dict[str, object] | None] = []

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        hiprune_ratio = kwargs.pop("hiprune_ratio", None)
        hiprune_prompt_ids = kwargs.pop("hiprune_prompt_ids", None)
        hiprune_config = kwargs.pop("hiprune_config", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if self.config.vision_config.model_type == "pixtral":
                return PixtralHFImagePixelInputs(
                    type="pixel_values_pixtral",
                    pixel_values=pixel_values,
                )

            expected_h = expected_w = self.config.vision_config.image_size
            return LlavaImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                hiprune_ratio=hiprune_ratio,
                hiprune_prompt_ids=hiprune_prompt_ids,
                hiprune_config=hiprune_config,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        if image_embeds is not None:
            if self.config.vision_config.model_type == "pixtral":
                raise ValueError("Pixtral-HF does not support image_embeds.")

            return LlavaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_tower: CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel,
        pixel_values: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        return vision_tower(
            pixel_values,
            feature_select_strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(
        self,
        inputs: LlavaImagePixelInputs | PixtralHFImagePixelInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        pixel_values = inputs["pixel_values"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(
        self,
        image_input: LlavaImageInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            self._hiprune_metadata = [None] * len(image_input["data"])
            return image_input["data"]

        num_images = len(image_input["pixel_values"])
        self._hiprune_metadata = [None] * num_images

        ratios: list[float | None] = [None] * num_images
        prompt_ids: list[torch.Tensor | None] = [None] * num_images
        configs: list[HipruneConfig | None] = [None] * num_images
        if image_input["type"] == "pixel_values":
            hp_field = image_input.get("hiprune_ratio")
            if hp_field is not None:
                for idx in range(num_images):
                    ratio = float(hp_field[idx])
                    ratios[idx] = ratio if ratio < 1.0 else None
            # HiPrune++ prompt ids, kept per image: a batch can span
            # multiple requests with different prompts.
            pid_field = image_input.get("hiprune_prompt_ids")
            if pid_field is not None:
                for idx in range(num_images):
                    row = pid_field[idx]
                    if row.numel():
                        prompt_ids[idx] = row
            # Packed method/knob config, kept per image: a batch can
            # span requests with different methods.
            cfg_field = image_input.get("hiprune_config")
            for idx in range(num_images):
                if ratios[idx] is not None:
                    row = cfg_field[idx] if cfg_field is not None else None
                    configs[idx] = unpack_hiprune_config(row)

        if any(r is not None for r in ratios):
            return self._process_image_input_hiprune(
                image_input["pixel_values"], ratios, prompt_ids, configs
            )

        image_features = self._process_image_pixels(image_input)

        if isinstance(image_features, torch.Tensor):
            return self.multi_modal_projector(image_features)

        feature_sizes = [image_feature.shape[0] for image_feature in image_features]

        image_embeds = self.multi_modal_projector(torch.cat(image_features))
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def _process_image_input_hiprune(
        self,
        pixel_values: torch.Tensor,
        ratios: list[float | None],
        prompt_ids: list[torch.Tensor | None] | None = None,
        configs: list[HipruneConfig | None] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Encode images one at a time, pruning with HiPrune where requested.

        Images are encoded individually because the score capture computes
        a dense attention softmax over the whole patch sequence, which
        must not span keys from other images — exactly like the HiPrune
        reference, which encodes one image at a time.

        For each pruned image, the selection (HiPrune's
        anchor/buffer/register, HyDART's anchor/buffer/diverse with
        ``HIPRUNE_METHOD=hydart``, or HiPrune++'s
        anchor/buffer/register/prompt with ``HIPRUNE_METHOD=hiprune_pp``)
        keeps exactly the token count the processor used for the
        placeholder run — ``compute_retained_tokens_count(num_tokens,
        ratio)``, plus the text-guided extra for HiPrune++ — and the
        metadata is stashed for the API layer.

        HiPrune++ compares the projected patch embeddings (LM input
        space, via ``multi_modal_projector``) against the mean LM
        embedding of the image's ``prompt_ids`` entry; a missing/empty
        prompt falls back to a zero text embedding (arbitrary text
        picks, count unchanged).

        DART (``HIPRUNE_METHOD=dart``) uses no vision-tower attention:
        the projected embeddings and the LM embeddings of the prompt run
        through the language model's first K decoder layers (aux pass,
        ``vllm/multimodal/dart_scoring.py``); the layer-K key norms and
        hidden states drive the official pivot + anti-duplication
        selection.

        Layer choices mirror the HiPrune authors' LLaVA-1.5 release: the
        object layer is layer 9 (1-based) of the CLIP tower, and the deep
        layer is the feature-select layer (-2, the last block vLLM loads
        since the tower is truncated there).
        """
        tower = self.vision_tower
        if not hasattr(tower, "forward_capturing_hiprune_scores"):
            raise NotImplementedError(
                "HiPrune token pruning is only wired into the CLIP vision "
                f"tower; got {type(tower).__name__}."
            )

        strategy = self.config.vision_feature_select_strategy
        if strategy != "default":
            raise NotImplementedError(
                "HiPrune for LLaVA assumes vision_feature_select_strategy="
                f"'default' (CLS dropped); got {strategy!r}."
            )

        vision_config = self.config.vision_config
        grid_w = grid_h = vision_config.image_size // vision_config.patch_size
        num_tokens = grid_w * grid_h

        num_loaded_layers = len(tower.vision_model.encoder.layers)
        object_layer_idx = LLAVA_OBJECT_LAYER - 1
        deep_layer_idx = num_loaded_layers - 1

        # HiPrune++: per-image mean LM embedding of the prompt tokens.
        def _text_embedding_for(idx: int) -> torch.Tensor | None:
            if prompt_ids is None or prompt_ids[idx] is None:
                return None
            ids = prompt_ids[idx].long()
            embed_tokens = self.language_model.model.embed_tokens
            return embed_tokens(ids).float().mean(dim=0)

        image_embeds_out: list[torch.Tensor] = []
        for idx, ratio in enumerate(ratios):
            pv = pixel_values[idx : idx + 1]
            if ratio is None:
                features = self._image_pixels_to_features(tower, pv)
                image_embeds_out.append(self.multi_modal_projector(features)[0])
                continue

            # Per-image method dispatch: images are encoded individually
            # anyway (see docstring), so a batch mixing methods across
            # requests just takes a different branch per image.
            cfg = (
                configs[idx]
                if configs is not None and configs[idx] is not None
                else unpack_hiprune_config(None)
            )
            method = cfg.method

            # NPrune: pure uniform-lattice sampling — no attention
            # capture, no scores, no prompt. Plain tower forward.
            if method == "nprune":
                features = self._image_pixels_to_features(tower, pv)
                embeds = self.multi_modal_projector(features)[0]
                assert embeds.shape[0] == num_tokens
                kept_idx, kept_mask = nprune_select(
                    grid_h, grid_w, cfg.stride, device=embeds.device
                )
                metadata = build_nprune_metadata(
                    kept_idx, kept_mask, grid_w, grid_h, cfg.stride
                )
                image_embeds_out.append(embeds[kept_mask])
                self._hiprune_metadata[idx] = metadata
                continue

            # Checkered: deterministic checkerboard — no attention
            # capture, no scores, no prompt. Plain tower forward.
            if method == "checkered":
                features = self._image_pixels_to_features(tower, pv)
                embeds = self.multi_modal_projector(features)[0]
                assert embeds.shape[0] == num_tokens
                kept_idx, kept_mask = checkered_select(
                    grid_h, grid_w, device=embeds.device
                )
                metadata = build_checkered_metadata(
                    kept_idx, kept_mask, grid_w, grid_h
                )
                image_embeds_out.append(embeds[kept_mask])
                self._hiprune_metadata[idx] = metadata
                continue

            # AnchorPrune: relevance-anchored contextual expansion
            # (prompt-aware). Importance prior = received attention
            # mass at the deepest loaded CLIP layer (the paper's
            # layer -2; vLLM loads CLIP only up to the feature-select
            # layer). Relevance and novelty run in LM space over the
            # projected embeds — the official Qwen adapter's signal,
            # applied here instead of the paper's extra CLIP text tower.
            if method == "anchorprune":
                features, scores_by_layer = (
                    tower.forward_capturing_hiprune_scores(
                        pv,
                        capture_layer_idxs=(deep_layer_idx,),
                        feature_select_strategy=strategy,
                    )
                )
                embeds = self.multi_modal_projector(features)[0]
                assert embeds.shape[0] == num_tokens
                # Per-key scores cover the full CLIP sequence; drop the
                # CLS key (index 0) to align with the 24x24 patches.
                importance = scores_by_layer[deep_layer_idx][1:]
                assert importance.shape[0] == num_tokens

                lm = self.language_model.model
                pid = prompt_ids[idx] if prompt_ids is not None else None
                if pid is not None and pid.numel():
                    prompt_embeds = lm.embed_tokens(
                        pid.long().to(embeds.device)
                    ).float()
                else:
                    prompt_embeds = None
                # Same count function as the processor's placeholder
                # sizing (generic ratio path).
                k_total = hiprune_retained_tokens_count(num_tokens, ratio)
                kept_idx, kept_mask, anchor_idx, expansion_idx = (
                    anchorprune_select(
                        embeds,
                        importance,
                        prompt_embeds,
                        k_total,
                        anchor_kmin=cfg.anchor_kmin,
                        tau=cfg.tau,
                    )
                )
                metadata = build_anchorprune_metadata(
                    anchor_idx,
                    expansion_idx,
                    kept_mask,
                    grid_w,
                    grid_h,
                    ratio,
                    anchor_kmin_used=anchorprune_resolve_kmin(
                        k_total, cfg.anchor_kmin
                    ),
                    tau=cfg.tau,
                    num_prompt_tokens=(
                        0 if prompt_embeds is None
                        else int(prompt_embeds.shape[0])
                    ),
                )
                image_embeds_out.append(embeds[kept_mask])
                self._hiprune_metadata[idx] = metadata
                continue

            if method == "dart":
                features = self._image_pixels_to_features(tower, pv)
                embeds = self.multi_modal_projector(features)[0]
                assert embeds.shape[0] == num_tokens

                lm = self.language_model.model
                pid = prompt_ids[idx] if prompt_ids is not None else None
                if pid is not None and pid.numel():
                    text_embeds = lm.embed_tokens(
                        pid.long().to(embeds.device)
                    ).to(embeds.dtype)
                else:
                    text_embeds = embeds.new_zeros((0, embeds.shape[-1]))
                seq_embeds = torch.cat([embeds, text_embeds], dim=0)
                positions = torch.arange(
                    seq_embeds.shape[0], device=embeds.device
                )
                dart_layer = get_dart_layer()
                hidden, key_l1 = dart_prefix_states_llama(
                    lm.layers, lm.norm, seq_embeds, positions, dart_layer
                )
                p_img, p_txt = cfg.pivot_image, cfg.pivot_text
                img_piv, txt_piv, diverse_idx, kept_mask, pivot_sim = (
                    dart_select(hidden, key_l1, num_tokens, ratio, p_img, p_txt)
                )
                metadata = build_dart_metadata(
                    img_piv,
                    diverse_idx,
                    kept_mask,
                    key_l1,
                    pivot_sim,
                    grid_w,
                    grid_h,
                    ratio,
                    pivot_image=p_img,
                    pivot_text=p_txt,
                    num_text_pivots=int(txt_piv.numel()),
                    dart_layer=dart_layer,
                )
                image_embeds_out.append(embeds[kept_mask])
                self._hiprune_metadata[idx] = metadata
                continue

            # HyDART needs only the object-layer scores (its diverse fill
            # uses embedding similarity, not deep attention).
            if method == "hydart":
                capture_layer_idxs: tuple[int, ...] = (object_layer_idx,)
            else:
                capture_layer_idxs = (object_layer_idx, deep_layer_idx)
            features, scores_by_layer = tower.forward_capturing_hiprune_scores(
                pv,
                capture_layer_idxs=capture_layer_idxs,
                feature_select_strategy=strategy,
            )
            embeds = self.multi_modal_projector(features)[0]
            assert embeds.shape[0] == num_tokens

            # Per-key scores cover the full CLIP sequence; drop the CLS
            # key (index 0) to align with the 24x24 patch tokens, exactly
            # like the reference (`attentions[layer].mean(1).mean(1)[0, 1:]`).
            shallow = scores_by_layer[object_layer_idx][1:]

            if method == "hydart":
                lambda_seed, lambda_pick = cfg.lambda_seed, cfg.lambda_pick
                anchor_idx, buffer_idx, diverse_idx, kept_mask, sim_stats = (
                    hydart_select(
                        shallow,
                        embeds,
                        num_tokens,
                        grid_w,
                        ratio,
                        lambda_seed=lambda_seed,
                        lambda_pick=lambda_pick,
                    )
                )
                metadata = build_hydart_metadata(
                    anchor_idx,
                    buffer_idx,
                    diverse_idx,
                    kept_mask,
                    shallow,
                    sim_stats,
                    grid_w,
                    grid_h,
                    ratio,
                    object_layer=LLAVA_OBJECT_LAYER,
                    lambda_seed=lambda_seed,
                    lambda_pick=lambda_pick,
                )
            elif method == "hiprune_pp":
                deep = scores_by_layer[deep_layer_idx][1:]
                text_emb = _text_embedding_for(idx)
                if text_emb is None:
                    text_emb = embeds.new_zeros(
                        embeds.shape[-1], dtype=torch.float32
                    )
                (
                    anchor_idx,
                    buffer_idx,
                    register_idx,
                    prompt_idx,
                    kept_mask,
                    text_similarity,
                ) = hiprune_pp_select(
                    shallow,
                    deep,
                    embeds,
                    text_emb,
                    num_tokens,
                    grid_w,
                    ratio,
                    beta=cfg.beta,
                )
                metadata = build_hiprune_pp_metadata(
                    anchor_idx,
                    buffer_idx,
                    register_idx,
                    prompt_idx,
                    kept_mask,
                    shallow,
                    deep,
                    text_similarity,
                    grid_w,
                    grid_h,
                    ratio,
                    object_layer=LLAVA_OBJECT_LAYER,
                    beta=cfg.beta,
                )
            else:
                deep = scores_by_layer[deep_layer_idx][1:]
                anchor_idx, buffer_idx, register_idx, kept_mask = hiprune_select(
                    shallow, deep, num_tokens, grid_w, ratio
                )
                metadata = build_hiprune_metadata(
                    anchor_idx,
                    buffer_idx,
                    register_idx,
                    kept_mask,
                    shallow,
                    deep,
                    grid_w,
                    grid_h,
                    ratio,
                    object_layer=LLAVA_OBJECT_LAYER,
                )
            image_embeds_out.append(embeds[kept_mask])
            self._hiprune_metadata[idx] = metadata

        return tuple(image_embeds_out)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            self.hiprune_metadata_per_item = []
            return []

        embeddings = self._process_image_input(image_input)
        # Per returned item: HiPrune metadata (None when the item was not
        # pruned). The model runner reads this right after embed_multimodal
        # to cache the metadata alongside the encoder outputs.
        self.hiprune_metadata_per_item = list(self._hiprune_metadata)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for LLaVA-1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"USER: <image>\\nWhat's the content of the image?\\nASSISTANT:"`.

        Tokenizer outputs:
        `[1, 3148, 1001, 29901, 29871, 32000, 29871, 13, 5618, 29915, 29879,
        278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 3148, 1001, 29901, 29871, 32000, ..., 32000, 29871, 13, 5618,
        29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566,
        29901]`.

        We insert 575 tokens so that including the original image token in the
        input, there are a total of 576 (24 * 24) image tokens, which
        corresponds to the number of image tokens inputted to the language
        model, i.e. the number of image tokens outputted by the visual encoder.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        Info:
            [`LlavaImageInputs`][vllm.model_executor.models.llava.LlavaImageInputs]
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="multi_modal_projector",
            tower_model="vision_tower",
        )

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        # LLaVA's vision encoder outputs one token per patch without
        # spatial merging or pixel shuffle
        return num_image_tokens

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        # LLaVA's MLP projector outputs the same number of tokens
        # as it receives from the vision encoder (1:1 mapping)
        return num_vision_tokens
