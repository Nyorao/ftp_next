"""Single-network inference from scalar volumes, without annotation files.

The legacy training inputs were quantized with Matplotlib's per-slice gray
colormap. Reproduce that operation before resizing, rather than introducing
an untested intensity-preprocessing change at inference time.
"""

import hashlib
from pathlib import Path

from matplotlib import colormaps, colors
import numpy as np
from PIL import Image
import SimpleITK as sitk
import torch
from torch import nn
from torch.nn import functional as F

from .config import ExperimentConfig
from .data import _image_shape
from .geometry import canonicalize, restore_native, sagittal_plan
from .models import UNet, haar_components


def image_geometry(image):
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Expected a scalar three-dimensional image")
    return {"size_xyz": list(image.GetSize()),
            "spacing_xyz_mm": list(image.GetSpacing()),
            "origin_xyz": list(image.GetOrigin()),
            "direction": list(image.GetDirection())}


def read_volume(path, geometry_record=None):
    """Read only an image; optionally repair known converted-image metadata.

    A repair is accepted only for the exact manifest path, shape and complete
    voxel hash. Raw source images normally use their own trustworthy header.
    No reference mask or class centroid is used to determine orientation.
    """
    path = Path(path).resolve()
    image = sitk.ReadImage(str(path))
    provided = image_geometry(image)
    array = sitk.GetArrayFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError("Input image contains non-finite intensities")
    digest = hashlib.sha256(array.tobytes()).hexdigest()
    verified = provided
    if geometry_record is not None:
        if Path(geometry_record["converted_image"]).resolve() != path:
            raise ValueError("Geometry record refers to a different image")
        if digest != geometry_record["converted_image_array_sha256"]:
            raise ValueError("Image voxel hash does not match the geometry manifest")
        verified = geometry_record["verified_geometry"]
        if list(array.shape[::-1]) != verified["size_xyz"]:
            raise ValueError("Image shape does not match verified geometry")
    plan = sagittal_plan(verified)
    return array, plan, {"input": str(path), "input_array_sha256": digest,
                         "provided_geometry": provided, "output_geometry": verified,
                         "geometry_source": "verified_image_manifest" if geometry_record else "input_header",
                         "sagittal_plan": plan, "annotation_files_read": 0}


def legacy_gray8(slice_ap_si):
    """Exactly reproduce PIL -> imsave(cmap='gray') -> grayscale conversion."""
    value = np.asarray(slice_ap_si)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("Expected a finite nonempty image slice")
    # PIL can change int16 to int32 / float64 to float32. That affects the
    # Normalize arithmetic and must agree with the original converter.
    value = np.asarray(Image.fromarray(value))
    return colormaps["gray"](colors.Normalize()(value), bytes=True)[..., 0].copy()


def prepare_slice(slice_ap_si, image_size):
    """Return (1,H,W), using the exact unaugmented training normalization."""
    gray = legacy_gray8(slice_ap_si)
    image = torch.from_numpy(gray.astype(np.float32))[None, None]
    image = F.interpolate(image, size=_image_shape(image_size), mode="bilinear", align_corners=False)[0]
    return (image - image.mean()) / image.std(unbiased=False).clamp_min(1e-6)


class SingleNetwork(nn.Module):
    """One selected backbone, including its original input-view transform."""

    def __init__(self, backbone, frequency_strength=0.0):
        super().__init__()
        self.backbone = backbone
        self.frequency_strength = frequency_strength

    def forward(self, images, position=None):
        if self.frequency_strength:
            _, high = haar_components(images)
            images = images + self.frequency_strength * high
        return self.backbone(images,**({} if position is None else {'position':position})).float().softmax(1)


def load_single_network(path, device, network=None):
    """Load one network from a format-3 checkpoint; no training system on GPU."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 3:
        raise ValueError("Expected a freqssl format-3 checkpoint")
    cfg = ExperimentConfig(**payload["config"]).validate()
    network = network or cfg.primary_network
    names = {"student1": ("students", 0), "student2": ("students", 1),
             "teacher1": ("teachers", 0), "teacher2": ("teachers", 1)}
    if network not in names:
        raise ValueError("Select one student or teacher, not an ensemble")
    collection, index = names[network]
    if index and not cfg.method.startswith("dual_"):
        raise ValueError(f"{network} is unavailable for {cfg.method}")
    architecture = UNet
    extra = {}
    if cfg.slice_context != "none":
        from .slice_context import ContextUNet
        architecture = ContextUNet
        extra["mode"] = cfg.slice_context
    if cfg.case_reference != 'none' or cfg.context_position != 'none':
        from .case_context import CaseContextUNet
        architecture = CaseContextUNet
        extra.update(case_reference=cfg.case_reference,context_position=cfg.context_position)
    if cfg.reference_condition != 'none':
        from .reference_conditioning import ReferenceContextUNet
        architecture = ReferenceContextUNet
        extra.update(reference_condition=cfg.reference_condition, reference_windows=cfg.reference_windows)
    if cfg.internal_text != "none":
        from .internal_text import TextContextUNet
        architecture = TextContextUNet
        extra["text_adapter_dim"] = cfg.text_adapter_dim
        if cfg.text_fusion_layout == "hierarchical":
            from .hierarchical_text import HierarchicalTextUNet
            architecture = HierarchicalTextUNet
            extra["text_layer_mode"] = cfg.text_layer_mode
        elif cfg.text_fusion_layout == "encoder":
            from .encoder_text import EncoderTextContextUNet
            architecture = EncoderTextContextUNet
            extra.update(text_feature_kind=cfg.text_feature_kind,
                         encoder_text_stage=cfg.encoder_text_stage)
        elif cfg.text_fusion_layout == "encoder_multi":
            from .dual_text import DualTextContextUNet
            architecture = DualTextContextUNet
            extra.update(encoder_text_stages=cfg.encoder_text_stages,
                text_routing_mode=cfg.text_routing_mode, text_residual_mode=cfg.text_residual_mode)
    if cfg.method == "dual_ordinal" and index == 1:
        from .ordinal_model import OrdinalUNet
        architecture = OrdinalUNet
    with torch.random.fork_rng(devices=[]):
        backbone = architecture(num_classes=cfg.num_classes, base_channels=cfg.base_channels,
                                max_channels=cfg.max_channels, **extra)
    prefix = f"{collection}.{index}."
    state = {key[len(prefix):]: value for key, value in payload["model"].items() if key.startswith(prefix)}
    backbone.load_state_dict(state, strict=True)
    if cfg.reference_condition != 'none':
        backbone.reference_teacher_path = cfg.reference_teacher
        backbone.reference_teacher_sha256 = cfg.reference_teacher_sha256
    strength = cfg.frequency_strength if cfg.method == "dual_frequency" and index == 1 else 0.0
    model = SingleNetwork(backbone, strength).requires_grad_(False).eval().to(device)
    return model, cfg, {"network": network, "checkpoint_step": payload["step"],
                        "parameters_loaded": sum(p.numel() for p in model.parameters()),
                        "backbones_loaded": 1, "method": cfg.method, "slice_context": cfg.slice_context,
                        "case_reference": cfg.case_reference, "context_position": cfg.context_position,
                        "reference_condition": cfg.reference_condition,
                        "reference_teacher_required": cfg.reference_condition != 'none',
                        "internal_text": cfg.internal_text,
                        "text_encoder_loaded": False,
                        "text_provenance": payload.get("text_provenance")}


@torch.inference_mode()
def predict_volume(model, array_zyx, plan, image_size, device, batch_size=8, reference_teacher=None):
    """Argmax, nearest resize and inverse layout; no structural postprocessing."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    canonical = canonicalize(array_zyx, plan)
    labels = np.empty(canonical.shape, dtype=np.uint8)
    was_training = model.training
    model.eval()
    reference_counter = None
    try:
        context = getattr(getattr(model, "backbone", None), "slice_context", "none") != "none"
        if context:
            from .context_cache import iter_cached_logits
            # Legacy PNG IDs follow increasing acquisition-array slice index.
            # Canonicalizing LR can reverse that index; adaptive weights are
            # ordered, so retain acquisition order while normalizing each plane.
            reverse_slices = 0 in plan["flip_canonical_axes"]
            ordered = canonical[::-1] if reverse_slices else canonical
            from .case_context import relative_position
            options = ({'read_position':lambda i:relative_position(i,len(ordered),reverse_slices)}
                       if getattr(model.backbone,'context_position','none')=='relative' else {})
            read_image = lambda index: prepare_slice(ordered[index], image_size)
            if getattr(model.backbone, 'reference_condition', 'none') != 'none':
                from .reference_inference import frozen_source_predictions, iter_reference_logits
                source = reference_teacher or model.backbone.reference_teacher_path
                probabilities, source_receipt = frozen_source_predictions(source, len(ordered), read_image,
                    device, model.backbone.reference_teacher_sha256, batch_size)
                reference_stats = {}
                iterator = iter_reference_logits(model.backbone, len(ordered), read_image,
                    lambda index: probabilities[index], device, batch_size, stats=reference_stats)
                model.reference_receipt = dict(source=source_receipt, target=reference_stats,
                    selection='maximum predicted canal area; middle-distance/index tie break',
                    source_checkpoint=source, backbones_required=2)
                from .reference_metrics import WorkCounter
                import time
                reference_counter = WorkCounter(model.backbone)
                target_started = time.monotonic()
            else:
                iterator = iter_cached_logits(model.backbone, len(ordered), read_image, device, batch_size, **options)
            for start, logits in iterator:
                if logits.shape[1] != 20 or not torch.isfinite(logits).all():
                    raise ValueError("Invalid context segmentation logits")
                prediction = F.interpolate(logits.argmax(1).float()[:, None],
                                           size=canonical.shape[1:], mode="nearest")
                values = prediction[:, 0].cpu().numpy().astype(np.uint8)
                if reverse_slices:
                    labels[len(labels)-start-len(logits):len(labels)-start] = values[::-1]
                else:
                    labels[start:start + len(logits)] = values
            if reference_counter is not None:
                if str(device).startswith('cuda'):
                    torch.cuda.synchronize(device)
                model.reference_receipt['target'].update(reference_counter.finish(), seconds=time.monotonic()-target_started)
                reference_counter = None
        for start in ([] if context else range(0, len(canonical), batch_size)):
            samples = canonical[start:start + batch_size]
            images = torch.stack([prepare_slice(value, image_size) for value in samples]).to(device)
            probabilities = model(images)
            if (probabilities.shape != (len(samples), 20, *_image_shape(image_size))
                    or not torch.isfinite(probabilities).all()):
                raise ValueError("Invalid segmentation probabilities")
            prediction = probabilities.argmax(1).float()[:, None]
            prediction = F.interpolate(prediction, size=canonical.shape[1:], mode="nearest")
            labels[start:start + len(samples)] = prediction[:, 0].cpu().numpy().astype(np.uint8)
    finally:
        if reference_counter is not None:
            reference_counter.finish()
        model.train(was_training)
    return restore_native(labels, plan)


def prediction_image(labels_zyx, geometry):
    if list(labels_zyx.shape[::-1]) != geometry["size_xyz"]:
        raise ValueError("Prediction and output geometry shapes differ")
    image = sitk.GetImageFromArray(labels_zyx.astype(np.uint8))
    image.SetSpacing(geometry["spacing_xyz_mm"])
    image.SetOrigin(geometry["origin_xyz"])
    image.SetDirection(geometry["direction"])
    return image
