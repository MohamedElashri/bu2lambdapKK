import argparse
import json
from pathlib import Path

import pandas as pd


def load_profile_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_combined_states(payload: dict) -> dict[str, dict]:
    return payload.get("combined", {})


def build_table(ll_data: dict[str, dict], dd_data: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for state in sorted(set(ll_data) | set(dd_data)):
        ll = ll_data.get(state, {})
        dd = dd_data.get(state, {})

        q0_ll = float(ll.get("q0", 0.0))
        q0_dd = float(dd.get("q0", 0.0))
        q0_combined = q0_ll + q0_dd

        rows.append(
            {
                "state": state,
                "z_ll": float(ll.get("z", 0.0)),
                "z_dd": float(dd.get("z", 0.0)),
                "q0_ll": q0_ll,
                "q0_dd": q0_dd,
                "q0_combined": q0_combined,
                "z_combined": q0_combined**0.5,
                "ll_best_fit_yield": float(ll.get("best_fit_yield", 0.0)),
                "dd_best_fit_yield": float(dd.get("best_fit_yield", 0.0)),
                "ll_best_fit_error": float(ll.get("best_fit_error", 0.0)),
                "dd_best_fit_error": float(dd.get("best_fit_error", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine LL and DD profile-likelihood significances at the q0 level."
    )
    parser.add_argument("--ll", type=Path, required=True, help="LL profile_significances.json")
    parser.add_argument("--dd", type=Path, required=True, help="DD profile_significances.json")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    args = parser.parse_args()

    ll_payload = load_profile_json(args.ll)
    dd_payload = load_profile_json(args.dd)

    df = build_table(extract_combined_states(ll_payload), extract_combined_states(dd_payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved combined profile significances to {args.output}")


if __name__ == "__main__":
    main()
