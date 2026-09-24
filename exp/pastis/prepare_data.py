"""One-time prep: convert raw PASTIS-R into the .pt format OlmoEarth's eval expects.

Uses OlmoEarth's own PASTISRProcessor, which:
  - imputes the 10 PASTIS S2 bands up to OlmoEarth's 13->12 band layout (by duplication),
  - aggregates the irregular time series into <=12 monthly averages,
  - splits each 128x128 tile into 4x 64x64 (unless --image_size 128),
  - maps PASTIS void class 19 -> ignore label,
  - writes pastis_r_{train,valid,test}/{s2_images,s1_images}/, months.pt, targets.pt.

--image_size picks the tile size, which is ALSO the processor's quadrant-split switch:
  64  (default) each raw 128x128 PASTIS patch is split into 4 tiles -> 5820 train samples.
      Matches every cached feature set and every row in results/pastis/*.csv.
  128 the raw patch is kept whole -> 1455 train samples, 4x the pixels each. This is the
      "pastis128" OlmoEarth dataset config (same as "pastis" but height_width=128).

Because the split changes the SAMPLE COUNT and per-sample indices, a 128 prep MUST go to its
own --output_dir: cached features are keyed by dataset index, so pointing 128 data at a
64-derived feature cache would silently misalign every sample. Metrics from the two preps are
also not comparable (different sample count, different per-sample task).

MEMORY: PASTISRProcessor.process() accumulates EVERY fold in RAM and then torch.cat's them,
so peak is roughly 2x the 28.7 GB dataset (~57 GB) regardless of --image_size. A 64 GB
allocation is OOM-killed (exit 137, "Killed", usually with an empty output dir and a
misleadingly clean-looking log). Give the job ~180 GB.

Run once (uses env_olmo, NOT env):
    source env_setup/env_olmo.sh
    python -u -m exp.pastis.prepare_data
    python -u -m exp.pastis.prepare_data --image_size 128 --output_dir data/pastis128_olmoearth
"""
import argparse

# Bootstrap MUST run before any olmoearth_pretrain import (see exp/common/olmo_bootstrap.py).
from exp.common import olmo_bootstrap  # type: ignore[import-not-found]
olmo_bootstrap.apply()  # before any olmoearth_pretrain.evals import

from olmoearth_pretrain.evals.datasets.pastis_processor import PASTISRProcessor  # noqa: E402

DATA_DIR = "data/PASTIS-R"            # raw PASTIS-R (DATA_S2/, DATA_S1A/, ANNOTATIONS/, metadata.geojson)
OUTPUT_DIR = "data/pastis_olmoearth"  # consumed by PASTISRDataset(path_to_splits=...)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=DATA_DIR, help="raw PASTIS-R root")
    p.add_argument("--output_dir", default=None,
                   help=f"where to write the processed splits (default {OUTPUT_DIR} for "
                        f"--image_size 64; REQUIRED for 128 so it cannot clobber the 64 prep)")
    p.add_argument("--image_size", type=int, default=64, choices=(64, 128),
                   help="64: split each raw 128x128 patch into 4 tiles (default). "
                        "128: keep the raw patch whole (OlmoEarth 'pastis128' config).")
    args = p.parse_args()

    if args.output_dir is None:
        if args.image_size != 64:
            raise SystemExit(
                "ERROR: --output_dir is required for --image_size 128 (refusing to default to "
                f"{OUTPUT_DIR}, which holds the 64x64 prep every cached feature set is keyed to).\n"
                "       e.g. --image_size 128 --output_dir data/pastis128_olmoearth")
        args.output_dir = OUTPUT_DIR

    print(f"prep: image_size={args.image_size} "
          f"({'no split, raw 128x128 patches' if args.image_size == 128 else 'split into 4x 64x64'})")
    print(f"      {args.data_dir} -> {args.output_dir}")
    processor = PASTISRProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        # resize_to_64 IS the quadrant-split switch (see PASTISRProcessor.process_sample).
        resize_to_64=args.image_size == 64,
    )
    processor.process()
    print(f"Done. Processed PASTIS-R written to {args.output_dir}/")


if __name__ == "__main__":
    main()
