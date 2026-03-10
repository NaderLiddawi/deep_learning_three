from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, stride=2),
            ConvBNReLU(out_ch, out_ch, stride=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBNReLU(out_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Make the decoder robust to non-divisible / odd input sizes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Classifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.encoder = nn.Sequential(
            ConvBNReLU(in_channels, 32, stride=1),
            DownBlock(32, 64),
            DownBlock(64, 128),
            DownBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        z = self.encoder(z)
        z = self.pool(z)
        return self.head(z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)


class Detector(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        self.stem = ConvBNReLU(in_channels, 16)
        self.down1 = DownBlock(16, 32)
        self.down2 = DownBlock(32, 64)
        self.down3 = DownBlock(64, 128)

        self.up3 = UpBlock(128, 64, 64)
        self.up2 = UpBlock(64, 32, 32)
        self.up1 = UpBlock(32, 16, 16)

        self.seg_head = nn.Conv2d(16, num_classes, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_hw = x.shape[-2:]
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        s0 = self.stem(z)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        d3 = self.up3(s3, s2)
        d2 = self.up2(d3, s1)
        d1 = self.up1(d2, s0)

        logits = self.seg_head(d1)
        depth = torch.sigmoid(self.depth_head(d1)).squeeze(1)

        # Enforce exact output shape match with the input, even for odd sizes.
        if logits.shape[-2:] != input_hw:
            logits = F.interpolate(logits, size=input_hw, mode="bilinear", align_corners=False)
        if depth.shape[-2:] != input_hw:
            depth = F.interpolate(depth.unsqueeze(1), size=input_hw, mode="bilinear", align_corners=False).squeeze(1)

        return logits, depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, depth = self(x)
        pred = logits.argmax(dim=1)
        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(model_name: str, with_weights: bool = False, **model_kwargs) -> nn.Module:
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)
    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: nn.Module) -> str:
    model_name = None
    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n
    if model_name is None:
        raise ValueError(f"Model type '{type(model)}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)
    return output_path


def calculate_model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("Classifier debug")
    x = torch.rand(batch_size, 3, 64, 64).to(device)
    clf = load_model("classifier").to(device)
    out = clf(x)
    print(f"  Input : {x.shape}")
    print(f"  Output: {out.shape}  (expect ({batch_size}, 6))")
    print(f"  Size  : {calculate_model_size_mb(clf):.2f} MB")

    print("=" * 50)
    print("Detector debug")
    x = torch.rand(batch_size, 3, 96, 128).to(device)
    det = load_model("detector").to(device)
    seg, dep = det(x)
    print(f"  Input    : {x.shape}")
    print(f"  Seg logits: {seg.shape}  (expect ({batch_size}, 3, 96, 128))")
    print(f"  Depth     : {dep.shape}  (expect ({batch_size}, 96, 128))")
    print(f"  Depth range: [{dep.min():.3f}, {dep.max():.3f}]  (expect [0,1])")
    print(f"  Size  : {calculate_model_size_mb(det):.2f} MB")


if __name__ == "__main__":
    debug_model()
