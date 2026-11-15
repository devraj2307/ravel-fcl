"""Dataset loading for the RAVEL-FCL experiments (MNIST / EMNIST-Balanced / CIFAR-10).

The execution environment intentionally has no NumPy, pandas or torchvision, so
this module implements the small amount of parsing needed to read the public
datasets directly with the Python standard library and PyTorch:

* MNIST and EMNIST are stored as gzipped IDX files.
* CIFAR-10 is stored as a Python pickle tar archive.

Downloaded/parsed tensors are cached as ``.pt`` files under ``data_root`` so
subsequent runs start immediately.  When an offline run is requested (or a
download fails), a deterministic synthetic image stream is generated instead;
this keeps ``main.py`` runnable in CI while preserving the real-data path for
experiments with network access.

All returned images are float32 in ``[-1, 1]`` with shape ``[N, C, 32, 32]``
(MNIST/EMNIST are zero-padded from 28x28 to 32x32 as specified in Section 6.3).
"""

from __future__ import annotations

import gzip
import math
import struct
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

DATASET_ALIASES = {
    "mnist": "mnist",
    "emnist": "emnist_balanced",
    "emnist-balanced": "emnist_balanced",
    "emnist_balanced": "emnist_balanced",
    "cifar10": "cifar10",
    "cifar-10": "cifar10",
    "cifar": "cifar10",
    "synthetic": "synthetic",
}

DATASET_INFO = {
    "mnist": {"num_classes": 10, "img_channels": 1, "img_size": 32},
    "emnist_balanced": {"num_classes": 47, "img_channels": 1, "img_size": 32},
    "cifar10": {"num_classes": 10, "img_channels": 3, "img_size": 32},
    "synthetic": {"num_classes": 10, "img_channels": 1, "img_size": 32},
}

# Anchor all dataset downloads to the repository root, independent of the
# directory from which the scripts are launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"

MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"


@dataclass
class DatasetBundle:
    """In-memory image dataset with cached per-class index lists."""

    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    num_classes: int
    img_channels: int
    img_size: int = 32
    synthetic: bool = False
    _train_index: Optional[Dict[int, torch.Tensor]] = field(default=None, repr=False)
    _test_index: Optional[Dict[int, torch.Tensor]] = field(default=None, repr=False)

    def indices(self, split: str = "train") -> Dict[int, torch.Tensor]:
        """Mapping class -> LongTensor of sample indices (cached)."""
        if split == "train":
            if self._train_index is None:
                self._train_index = _build_class_index(self.train_y)
            return self._train_index
        if split == "test":
            if self._test_index is None:
                self._test_index = _build_class_index(self.test_y)
            return self._test_index
        raise ValueError("split must be 'train' or 'test'")

    def subset(self, split: str, classes: Iterable[int],
               max_per_class: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the samples belonging to ``classes`` from one split."""
        x, y = (self.train_x, self.train_y) if split == "train" else (self.test_x, self.test_y)
        pieces: List[torch.Tensor] = []
        for c in sorted({int(c) for c in classes}):
            idx = self.indices(split).get(c)
            if idx is None:
                continue
            if max_per_class is not None and idx.numel() > int(max_per_class):
                idx = idx[: int(max_per_class)]
            pieces.append(idx)
        if not pieces:
            return x[:0], y[:0]
        sel = torch.cat(pieces)
        return x[sel], y[sel]


def _build_class_index(labels: torch.Tensor) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for c in labels.unique().tolist():
        out[int(c)] = (labels == int(c)).nonzero(as_tuple=False).flatten()
    return out


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------
def load_dataset(
    name: str,
    root: Union[str, Path] = DEFAULT_DATA_ROOT,
    allow_download: bool = True,
    synthetic: bool = False,
    seed: int = 0,
    synthetic_samples_per_class: int = 200,
    verbose: bool = True,
) -> DatasetBundle:
    """Load one of ``mnist``, ``emnist_balanced``, ``cifar10`` or ``synthetic``.

    ``synthetic=True`` (or ``name='synthetic'``) bypasses all I/O.  If a real
    dataset is unavailable and ``allow_download`` is false, a synthetic bundle
    with the requested dataset's shape is returned (the caller can inspect
    ``bundle.synthetic`` and log a warning).
    """
    key = DATASET_ALIASES.get(str(name).lower())
    if key is None:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(DATASET_ALIASES)}"
        )
    root = Path(root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if synthetic or key == "synthetic":
        info = DATASET_INFO["mnist"]
        return make_synthetic_dataset(
            num_classes=info["num_classes"],
            img_channels=info["img_channels"],
            img_size=info["img_size"],
            samples_per_class=synthetic_samples_per_class,
            seed=seed,
            name="synthetic",
        )

    info = DATASET_INFO[key]
    try:
        if key == "mnist":
            bundle = _load_mnist(root, allow_download, verbose)
        elif key == "emnist_balanced":
            bundle = _load_emnist_balanced(root, allow_download, verbose)
        else:
            bundle = _load_cifar10(root, allow_download, verbose)
        return bundle
    except Exception as exc:  # offline / download failure -> deterministic fallback
        if verbose:
            print(f"[data] could not load {key} ({type(exc).__name__}: {exc}); "
                  f"falling back to synthetic images")
        return make_synthetic_dataset(
            num_classes=info["num_classes"],
            img_channels=info["img_channels"],
            img_size=info["img_size"],
            samples_per_class=synthetic_samples_per_class,
            seed=seed,
            name=f"{key}_synthetic",
        )


# --------------------------------------------------------------------------------------
# Download / cache helpers
# --------------------------------------------------------------------------------------
def _download(url: str, path: Path, verbose: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if verbose:
        print(f"[data] downloading {url} -> {path}")
    request = urllib.request.Request(url, headers={"User-Agent": "ravel-fcl/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        next_report = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if verbose and total and downloaded >= next_report:
                pct = 100.0 * downloaded / total
                print(f"[data]   {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)")
                next_report += max(total // 10, 1)
    tmp.replace(path)


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")
    except Exception:  # trusted local cache written by this module
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------------------
# IDX parsing (MNIST / EMNIST)
# --------------------------------------------------------------------------------------
def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    if len(data) < 16:
        raise ValueError(f"truncated IDX file: {path}")
    magic, n, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"bad IDX image magic {magic} in {path}")
    expected = n * rows * cols
    raw = torch.frombuffer(bytearray(data[16:16 + expected]), dtype=torch.uint8)
    if raw.numel() != expected:
        raise ValueError(f"truncated IDX payload in {path}")
    return raw.view(n, 1, rows, cols).float()


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        data = f.read()
    if len(data) < 8:
        raise ValueError(f"truncated IDX label file: {path}")
    magic, n = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"bad IDX label magic {magic} in {path}")
    return torch.frombuffer(bytearray(data[8:8 + n]), dtype=torch.uint8).long()


def _prepare_images(x: torch.Tensor, img_size: int = 32) -> torch.Tensor:
    """Normalise to [-1, 1] and zero-pad 28x28 -> 32x32."""
    x = x.float() / 255.0
    if x.size(1) == 1 and x.size(-1) == 28 and img_size == 32:
        x = F.pad(x, (2, 2, 2, 2))
    return x.mul(2.0).sub(1.0)


def _load_mnist(root: Path, allow_download: bool, verbose: bool) -> DatasetBundle:
    cache = root / "mnist" / "mnist_32.pt"
    if cache.exists():
        payload = _torch_load(cache)
        return DatasetBundle(name="mnist", **payload)

    raw_dir = root / "mnist" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in MNIST_FILES:
        path = raw_dir / filename
        if not path.exists():
            if not allow_download:
                raise FileNotFoundError(f"{path} not found (downloads disabled)")
            _download(MNIST_URL + filename, path, verbose=verbose)

    train_x = _read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    train_y = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    test_x = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    test_y = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    payload = dict(
        train_x=_prepare_images(train_x, 32),
        train_y=train_y,
        test_x=_prepare_images(test_x, 32),
        test_y=test_y,
        num_classes=10,
        img_channels=1,
        img_size=32,
        synthetic=False,
    )
    if verbose:
        print(f"[data] MNIST: {payload['train_x'].shape[0]} train / "
              f"{payload['test_x'].shape[0]} test")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return DatasetBundle(name="mnist", **payload)


def _load_emnist_balanced(root: Path, allow_download: bool, verbose: bool) -> DatasetBundle:
    cache = root / "emnist_balanced" / "emnist_balanced_32.pt"
    if cache.exists():
        payload = _torch_load(cache)
        return DatasetBundle(name="emnist_balanced", **payload)

    emnist_dir = root / "emnist_balanced"
    emnist_dir.mkdir(parents=True, exist_ok=True)
    members = {
        "train_x": "emnist-balanced-train-images-idx3-ubyte.gz",
        "train_y": "emnist-balanced-train-labels-idx1-ubyte.gz",
        "test_x": "emnist-balanced-test-images-idx3-ubyte.gz",
        "test_y": "emnist-balanced-test-labels-idx1-ubyte.gz",
    }
    paths: Dict[str, Path] = {}
    for key, basename in members.items():
        direct = emnist_dir / basename
        if not direct.exists():
            archive = emnist_dir / "gzip.zip"
            if not archive.exists():
                if not allow_download:
                    raise FileNotFoundError(f"{direct} not found (downloads disabled)")
                _download(EMNIST_URL, archive, verbose=verbose)
            if not direct.exists():
                with zipfile.ZipFile(archive) as zf:
                    match = next((n for n in zf.namelist()
                                  if n.endswith("/" + basename) or n.endswith(basename)), None)
                    if match is None:
                        raise FileNotFoundError(f"{basename} not found in {archive}")
                    if verbose:
                        print(f"[data] extracting {match}")
                    zf.extract(match, emnist_dir / "_extract")
                    direct = emnist_dir / "_extract" / match
        paths[key] = direct

    train_x = _read_idx_images(paths["train_x"])
    train_y = _read_idx_labels(paths["train_y"])
    test_x = _read_idx_images(paths["test_x"])
    test_y = _read_idx_labels(paths["test_y"])
    # EMNIST IDX images are stored transposed relative to MNIST; rotate them
    # into the conventional upright orientation.  Both orientations are valid
    # streams, but rotating keeps the visualisation/wasserstein probes sane.
    train_x = train_x.transpose(-1, -2)
    test_x = test_x.transpose(-1, -2)
    payload = dict(
        train_x=_prepare_images(train_x, 32),
        train_y=train_y,
        test_x=_prepare_images(test_x, 32),
        test_y=test_y,
        num_classes=47,
        img_channels=1,
        img_size=32,
        synthetic=False,
    )
    if verbose:
        print(f"[data] EMNIST-Balanced: {payload['train_x'].shape[0]} train / "
              f"{payload['test_x'].shape[0]} test")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return DatasetBundle(name="emnist_balanced", **payload)


# --------------------------------------------------------------------------------------
# CIFAR-10
# --------------------------------------------------------------------------------------
def _load_cifar10(root: Path, allow_download: bool, verbose: bool) -> DatasetBundle:
    cache = root / "cifar10" / "cifar10_32.pt"
    if cache.exists():
        payload = _torch_load(cache)
        return DatasetBundle(name="cifar10", **payload)

    base = root / "cifar10"
    extracted = base / "cifar-10-batches-bin"
    if not extracted.exists():
        archive = base / "cifar-10-binary.tar.gz"
        if not archive.exists():
            if not allow_download:
                raise FileNotFoundError(f"{archive} not found (downloads disabled)")
            _download(CIFAR10_URL, archive, verbose=verbose)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(base)
        extracted = base / "cifar-10-batches-bin"

    def read_batches(names: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parse the NumPy-free CIFAR-10 binary format (3073 bytes/record)."""
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for filename in names:
            with (extracted / filename).open("rb") as f:
                raw = f.read()
            buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            if buf.numel() % 3073 != 0:
                raise ValueError(f"malformed CIFAR-10 batch {filename}")
            records = buf.view(-1, 3073)
            ys.append(records[:, 0].long())
            xs.append(records[:, 1:].view(-1, 3, 32, 32))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = read_batches(
        [f"data_batch_{i}.bin" for i in range(1, 6)]
    )
    test_x, test_y = read_batches(["test_batch.bin"])
    payload = dict(
        train_x=_prepare_images(train_x, 32),
        train_y=train_y,
        test_x=_prepare_images(test_x, 32),
        test_y=test_y,
        num_classes=10,
        img_channels=3,
        img_size=32,
        synthetic=False,
    )
    if verbose:
        print(f"[data] CIFAR-10: {payload['train_x'].shape[0]} train / "
              f"{payload['test_x'].shape[0]} test")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return DatasetBundle(name="cifar10", **payload)


# --------------------------------------------------------------------------------------
# Deterministic synthetic fallback
# --------------------------------------------------------------------------------------
def make_synthetic_dataset(
    num_classes: int = 10,
    img_channels: int = 1,
    img_size: int = 32,
    samples_per_class: int = 200,
    test_samples_per_class: Optional[int] = None,
    seed: int = 0,
    name: str = "synthetic",
) -> DatasetBundle:
    """Class-conditional sinusoidal patterns with Gaussian noise.

    Each class receives a unique spatial frequency / orientation and (for RGB)
    colour, so a small classifier can learn a non-trivial mapping while the
    generator still has structured targets.
    """
    test_samples_per_class = test_samples_per_class or max(20, samples_per_class // 4)
    coords = torch.linspace(-1.0, 1.0, img_size)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    bases: List[torch.Tensor] = []
    for c in range(int(num_classes)):
        theta = math.pi * (c + 1) / (num_classes + 1)
        freq = 2.0 + 3.0 * ((c % 5) + 1) / 5.0
        phase = 2.0 * math.pi * (c % 7) / 7.0
        wave = torch.sin(freq * (xx * math.cos(theta) + yy * math.sin(theta)) + phase)
        base = wave.unsqueeze(0).repeat(img_channels, 1, 1)
        if img_channels >= 3:
            base[0] = wave
            base[1] = torch.cos(freq * (xx * math.sin(theta) - yy * math.cos(theta)) + phase)
            base[2] = 0.5 * (wave + base[1])
        bases.append(base)

    def build(n_per_class: int, split_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(split_seed)
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        for c, base in enumerate(bases):
            noise = torch.randn(n_per_class, img_channels, img_size, img_size,
                                generator=g) * 0.18
            xs.append((base.unsqueeze(0) + noise).clamp(-1.0, 1.0))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        return torch.cat(xs), torch.cat(ys)

    train_x, train_y = build(samples_per_class, seed)
    test_x, test_y = build(test_samples_per_class, seed + 10_007)
    return DatasetBundle(
        name=name,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        num_classes=int(num_classes),
        img_channels=int(img_channels),
        img_size=int(img_size),
        synthetic=True,
    )


# --------------------------------------------------------------------------------------
# Batch sampling
# --------------------------------------------------------------------------------------
class ClassBatchSampler:
    """Samples balanced minibatches restricted to a client's current classes.

    When ``omega != 1`` the sampling probability of a previously seen class is
    multiplied by ``omega`` (Stress Axis 1, Section 3.1); unseen classes keep
    weight 1.0.  With ``omega=0`` only the newly introduced classes are drawn
    (uniform fallback if the task contains no new classes).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seed: int = 0,
    ) -> None:
        self.x = x
        self.y = y
        self.index = _build_class_index(y)
        self._generator = torch.Generator().manual_seed(int(seed))

    def sample_batch(
        self,
        classes: Sequence[int],
        batch_size: int,
        previous_classes: Optional[Sequence[int]] = None,
        omega: float = 1.0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        classes = [int(c) for c in classes if int(c) in self.index]
        if not classes:
            return None
        previous = {int(c) for c in (previous_classes or ())}
        if omega != 1.0:
            weights = torch.tensor(
                [float(omega) if c in previous else 1.0 for c in classes],
                dtype=torch.float32,
            )
            if float(weights.sum()) <= 0.0:  # omega=0 and no unseen class
                weights = torch.ones(len(classes))
            labels = torch.multinomial(
                weights, int(batch_size), replacement=True, generator=self._generator
            )
            chosen = torch.tensor(classes, dtype=torch.long)[labels]
        else:
            chosen = torch.tensor(classes, dtype=torch.long)[
                torch.randint(len(classes), (int(batch_size),), generator=self._generator)
            ]

        index = torch.empty(int(batch_size), dtype=torch.long)
        for c in chosen.unique().tolist():
            mask = chosen == int(c)
            pool = self.index[int(c)]
            picks = torch.randint(
                pool.numel(), (int(mask.sum()),), generator=self._generator
            )
            index[mask] = pool[picks]
        return self.x[index], self.y[index]

    def make_batches(
        self,
        classes: Sequence[int],
        batch_size: int,
        num_batches: int,
        previous_classes: Optional[Sequence[int]] = None,
        omega: float = 1.0,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(max(0, int(num_batches))):
            batch = self.sample_batch(classes, batch_size, previous_classes, omega)
            if batch is not None:
                batches.append(batch)
        return batches
