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
    args = parser.parse_args()

    applicants = adms.load_applicants()
    output = {
        "model_1": adms.model_1(applicants, k=args.k),
        "model_2": adms.model_2(applicants, k=args.k),
        "model_3": adms.model_3(applicants, k=args.k),
        "model_4": adms.model_4(applicants, k=args.k),
        "model_5": adms.model_5(applicants, k=args.k),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
