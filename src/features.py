"""Optional torque z-score and CUSUM features from a frozen reference."""
from __future__ import annotations

import numpy as np
import pandas as pd

REFERENCE_DAYS = 21      # the window each unit is anchored on
CUSUM_K = 0.5            # slack, in reference SDs
SOURCE = "mixing_torque_nm"

FROZEN = ["torque_z_ref", "torque_cusum_up", "torque_cusum_down"]


def add_frozen_features(df: pd.DataFrame, col: str = SOURCE,
                        k: float = CUSUM_K,
                        reference_days: int = REFERENCE_DAYS
                        ) -> pd.DataFrame:
    """Attach the three columns. Row order is preserved."""
    if all(c in df.columns for c in FROZEN):
        return df

    d = df.sort_values(["unit", "ts"])
    z = np.full(len(d), np.nan)
    up = np.full(len(d), np.nan)
    dn = np.full(len(d), np.nan)

    pos = 0
    for _, g in d.groupby("unit", sort=False):
        n = len(g)
        end = g.ts.iloc[0] + pd.Timedelta(days=reference_days)
        ref = g[col][g.ts < end]
        mu, sd = ref.mean(), ref.std(ddof=1)
        zi = ((g[col].to_numpy() - mu) / sd) if sd and sd > 0 \
            else np.full(n, np.nan)
        z[pos:pos + n] = zi

        # A blank reading contributes nothing but must not reset the
        # tally: the previous value is carried through.
        cp = cm = 0.0
        ru, rd = np.empty(n), np.empty(n)
        for i, v in enumerate(zi):
            if not np.isnan(v):
                cp = max(0.0, cp + v - k)
                cm = max(0.0, cm - v - k)
            ru[i], rd[i] = cp, cm
        up[pos:pos + n], dn[pos:pos + n] = ru, rd
        pos += n

    out = df.copy()
    built = pd.DataFrame({"torque_z_ref": z, "torque_cusum_up": up,
                          "torque_cusum_down": dn}, index=d.index)
    for c in FROZEN:
        out[c] = built[c].reindex(df.index)
    return out


# =================================================================

def main() -> None:
    """Materialise the clean data plus the optional temporal columns."""
    import sys

    import pandas as pd

    from paths import CLEAN_PARQUET, MODEL_INPUT

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("  MODEL INPUT - clean data plus the frozen-reference features")
    print("=" * 70)
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"  read  {CLEAN_PARQUET.name}   {df.shape[0]:,} x {df.shape[1]}")
    out = add_frozen_features(df)
    print(f"  built {', '.join(FROZEN)}")
    for c in FROZEN:
        v = out[c]
        print(f"    {c:20s} nulls {v.isna().mean() * 100:5.2f} %   "
              f"min {v.min():9.3f}   max {v.max():11.1f}")
    out.to_parquet(MODEL_INPUT, index=False)
    mb = MODEL_INPUT.stat().st_size / 1e6
    print()
    print(f"  wrote {MODEL_INPUT.name}   {out.shape[0]:,} x "
          f"{out.shape[1]}   {mb:.1f} MB")
    print()


if __name__ == "__main__":
    main()
