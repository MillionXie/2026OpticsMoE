"""Run the same full-record replay schedule with a sequential D2NN."""

from .train_lifelong_moe import main


if __name__ == "__main__":
    main("d2nn")
