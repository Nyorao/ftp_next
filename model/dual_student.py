"""Independent LF/APGM and HF/PSGM students with one LF EMA teacher."""

import torch

from .mean_teacher import FTPStudent, MeanTeacher


class PromptStudent(FTPStudent):
    def __init__(self, text_features, class_embeddings, prompt, base_channels=32, enabled=True):
        if prompt not in ("apgm", "psgm"):
            raise ValueError("prompt must be apgm or psgm")
        super().__init__(text_features, class_embeddings, base_channels,
                         use_apgm=enabled and prompt == "apgm",
                         use_psgm=enabled and prompt == "psgm")
        self.branch = "lf" if prompt == "apgm" else "hf"

    def forward(self, view, return_plain=True):
        if view.ndim != 4 or view.shape[1] != 2 or min(view.shape[-2:]) < 64:
            raise ValueError("A student needs [batch, 2, height>=64, width>=64]")
        skips = self.unet.encode(view)
        bottleneck = skips[-1]
        if self.apgm is not None:
            batch, channels, height, width = bottleneck.shape
            tokens = bottleneck.flatten(2).transpose(1, 2)
            tokens = self.apgm(tokens, self.text_features.expand(batch, -1, -1))
            bottleneck = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        if self.psgm is not None:
            bottleneck = self.psgm(bottleneck)
        result = {f"{self.branch}_prompt": self.unet.decode(skips, bottleneck)}
        if return_plain:
            # No prompt means one decoder call, including one BN update.
            result[f"{self.branch}_plain"] = (self.unet.decode(skips)
                if self.apgm is not None or self.psgm is not None
                else result[f"{self.branch}_prompt"])
        return result


class SingleTeacherStudents(MeanTeacher):
    """EMA reads only S1. Feedback into S1 is an explicit loss, not weight mixing.

    The optional S2 is initialized in an isolated RNG context. Creating it does
    not perturb the initialization/augmentation stream of the anchor control.
    """

    def __init__(self, text_features, class_embeddings, base_channels=32,
                 ema_decay=0.99, dual=True, secondary_seed=43,
                 use_apgm=True, use_psgm=True):
        primary = PromptStudent(text_features, class_embeddings, "apgm", base_channels, use_apgm)
        super().__init__(primary, ema_decay)
        self.student2 = None
        if dual:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(secondary_seed)
                self.student2 = PromptStudent(text_features, class_embeddings, "psgm", base_channels, use_psgm)

    def forward(self, low_view, high_view, return_plain=True):
        outputs = self.student(low_view, return_plain=return_plain)
        if self.student2 is not None:
            outputs.update(self.student2(high_view, return_plain=return_plain))
        return outputs

    @torch.no_grad()
    def teacher_predictions(self, low_view, high_view=None):
        return self.teacher(low_view, return_plain=False)
