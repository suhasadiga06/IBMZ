"""Convert the Zhang et al. (2023) battery charging snippets into one plain .npz per brand.

    python -m chargeguard.scripts.convert_vehicle --src ~/evdata/battery_brand3 --out ~/evdata/vehicle/brand3.npz

Each snippet file was written with torch.save: a zip holding one pickle of
(array float64 (128, 8), {'label', 'car', 'charge_segment', 'mileage'}). The pickles hold
only numpy objects (and, in brand 2, an OrderedDict), so they are read with a restricted
unpickler that allows exactly the globals those need (nothing else can run), and neither
torch nor the notebook container is needed. numpy keeps the stored byte order ('<f8'), so the values come out
right on big-endian s390x; they are converted to native float32 before saving (cell
voltages keep ~1e-6 V resolution and timestamps are whole seconds below 2**24, so nothing
that matters is lost, and brand 2's ~194k snippets fit in memory on the 8 GB VM).
"""
from __future__ import annotations

import argparse
import io
import pickle
import time
import zipfile
from pathlib import Path

import numpy as np

# Every global these files use (checked with pickletools on all brand-3 and a sample of brand-2 files).
ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
           ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
           ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode"), ("collections", "OrderedDict")}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed")
        return super().find_class(module, name)

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("tensor storages are not expected in these files")


def read(path: Path):
    """Unpickle one torch.save file (zip format) without torch."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("/data.pkl"))
        return _Restricted(io.BytesIO(z.read(name)), encoding="latin1").load()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="extracted battery_brandN folder")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.src).expanduser()
    columns = [str(c) for c in read(src / "column.pkl")]
    files = sorted((p for p in src.rglob("*.pkl") if p.name != "column.pkl"), key=lambda p: (p.parent.name, p.stem))
    t = time.time()
    X = np.empty((len(files), 128, len(columns)), np.float32)
    car, label, seg, mileage, name, bad = [], [], [], [], [], []
    for p in files:
        try:
            arr, meta = read(p)
        except Exception as e:  # noqa: BLE001 - report and skip unreadable files
            bad.append(f"{p.name}: {e}")
            continue
        arr = np.asarray(arr)  # stored as '<f8'; numpy honours that byte order on s390x
        if arr.shape != (128, len(columns)):
            bad.append(f"{p.name}: shape {arr.shape}")
            continue
        X[len(car)] = arr  # cast to native float32
        car.append(int(meta["car"]))
        label.append(str(meta["label"]))
        seg.append(str(meta["charge_segment"]))
        mileage.append(float(meta["mileage"]))
        name.append(f"{p.parent.name}/{p.stem}")
    X = X[:len(car)]
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, columns=np.array(columns), car=np.array(car), label=np.array(label),
                        charge_segment=np.array(seg), mileage=np.array(mileage), snippet=np.array(name))
    print(f"{len(X):,} snippets from {len(set(car))} cars in {time.time() - t:.0f}s -> {out}  ({len(bad)} skipped)")
    for b in bad[:10]:
        print("  skipped", b)
    print("columns:", columns)
    for j, c in enumerate(columns):
        v = X[:, :, j]
        print(f"  {c:16s} min {v.min():12.4f}  p1 {np.percentile(v, 1):12.4f}  median {np.median(v):12.4f}  "
              f"p99 {np.percentile(v, 99):12.4f}  max {v.max():12.4f}")


if __name__ == "__main__":
    main()
