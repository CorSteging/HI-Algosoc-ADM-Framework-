import argparse
import json

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
    args = parser.parse_args()

    if args.model:
        output = {args.model: adms.run_model(args.model, k=args.k)}
    else:
        output = adms.run_all_models(k=args.k)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
