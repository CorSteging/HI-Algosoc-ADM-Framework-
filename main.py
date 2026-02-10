import argparse
import json
from pathlib import Path

import adms


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all ADM models and output top-k.")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of top applicants to return per model.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(adms.MODEL_REGISTRY.keys()),
        help="Run a single model instead of all models.",
    )
    parser.add_argument(
        "--input",
        default=str(adms.DATA_PATH),
        help="Path to the input JSON file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.model:
        output = {args.model: adms.run_model(args.model, k=args.k, path=input_path)}
    else:
        output = adms.run_all_models(k=args.k, path=input_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
