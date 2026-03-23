import argparse
import pandas as pd

from pipeline.classification import run_full_experiment_resumable, setup_dataset
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run resumable classification experiment with weak/gold labels."
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model name or path passed to the classification pipeline."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default="df_fse.csv",
        help="Input CSV file."
    )
    parser.add_argument(
        "--weak_label_col",
        type=str,
        default="our_bronchio",
        help="Column containing weak labels."
    )
    parser.add_argument(
        "--gold_label_col",
        type=str,
        default="bronchiolite",
        help="Column containing gold labels."
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="testo_clean",
        help="Text column used for classification."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="classification_clean",
        help="Base experiment name used for saved results."
    )
    parser.add_argument(
        "--weak_percentages",
        type=int,
        nargs="+",
        default=[100],
        help="List of weak-label percentages to evaluate, e.g. --weak_percentages 0 20 40 60 80 100"
    )
    parser.add_argument(
        "--n_reps",
        type=int,
        default=1,
        help="Number of repetitions."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Apply text cleaning in setup_dataset."
    )
    parser.add_argument(
        "--no_clean",
        action="store_false",
        dest="clean",
        help="Disable text cleaning in setup_dataset."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.input_csv)

    dataset, df_bal = setup_dataset(
        df,
        text_col=args.text_col,
        gold_label_col=args.gold_label_col,
        weak_label_col=args.weak_label_col,
        clean=args.clean
    )

    all_results = run_full_experiment_resumable(
        df_bal=df_bal,
        model=args.model,
        text_col=args.text_col,
        weak_label_col=args.weak_label_col,
        gold_label_col=args.gold_label_col,
        base_experiment_name=args.experiment_name,
        dataset=dataset,
        weak_percentages=args.weak_percentages,
        n_reps=args.n_reps
    )

    print("Finished!")


if __name__ == "__main__":
    main()