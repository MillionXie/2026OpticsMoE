import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.parse_args()
    raise SystemExit("Expert ablation is reserved for the next dataset-switching phase.")


if __name__ == "__main__":
    main()
