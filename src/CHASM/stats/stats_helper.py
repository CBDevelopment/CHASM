from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from chasm.data.folder_utils import *
import matplotlib.pyplot as plt


class StatisticsHelper:
    def __init__(self):
        pass

    @staticmethod
    def plot_histogram(
        data,
        bins=100,
        title="Histogram",
        xlabel="Value",
        ylabel="Frequency",
        save_path=None,
    ):
        """
        Make a histogram of the data with the specified number of bins.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=bins, range=(0, 1), alpha=0.5, color="blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)

    @staticmethod
    def determine_thresholds(masks, min_threshold=0.0, max_threshold_count=200):
        """
        Determine a set of thresholds to test based on the underlying data using a binning procedure.

        Inputs:
        - masks: list of model outputs to be analyzed
        - min_threshold: minimum threshold value to consider
        - max_threshold_count: maximum number of thresholds to generate

        Returns:
        - flattened, thresholds: a tuple containing the flattened data and the generated thresholds
        """
        # Flatten all model outputs into a 1D array
        flattened = np.concatenate([output.flatten() for output in masks])

        # Filter values to only consider those >= min_threshold
        flattened = flattened[flattened >= min_threshold]

        # If no values above min_threshold, return just the min_threshold
        if len(flattened) == 0:
            return [min_threshold]

        # Use percentiles to create a reasonable number of thresholds
        # This gives more resolution in areas with more data points
        num_percentiles = min(max_threshold_count, len(np.unique(flattened)))

        if num_percentiles <= 1:
            return [min_threshold]

        # Generate percentiles from min_threshold to max value
        percentiles = np.linspace(0, 100, num_percentiles)
        thresholds = np.percentile(flattened, percentiles)

        # Remove duplicates
        thresholds = np.unique(thresholds)

        print(
            f"Generated {len(thresholds)} thresholds between {thresholds[0]:.4f} and {thresholds[-1]:.4f}"
        )

        return flattened, thresholds

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        # Ensure inputs are binary integers
        y_true = np.array(y_true, dtype=np.uint8)
        y_pred = np.array(y_pred, dtype=np.uint8)

        intersection = y_true & y_pred
        union = y_true | y_pred
        tp = np.sum(intersection)
        fp = np.sum(y_pred & (1 - y_true))
        fn = np.sum((1 - y_pred) & y_true)
        tn = np.sum((1 - y_pred) & (1 - y_true))

        probability_of_detection = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tss = probability_of_detection - false_alarm_rate
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {
            "tss": tss,
            "accuracy": accuracy,
            "iou": iou,
            "dice": dice,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    @staticmethod
    def test_single_threshold(threshold, predictions, ground_truth):
        try:
            binary_predictions = np.concatenate(
                [(p > threshold).flatten() for p in predictions]
            )

            metrics = StatisticsHelper.calculate_metrics(
                ground_truth, binary_predictions
            )
            metrics["threshold"] = threshold
            return metrics

        except ZeroDivisionError as e:
            print(f"Error processing threshold {threshold}: {e}")
            return None

    @staticmethod
    def optimize_and_evaluate(
        pred_folder,
        gt_folder,
        training_months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        testing_months=[11, 12],
        metric_weights={"tss": 0.3333, "accuracy": 0.3333, "iou": 0.3333},
    ):
        """
        Optimize threshold on training months, then evaluate on testing months.

        Args:
            pred_folder: Path to predictions folder
            gt_folder: Path to ground truth folder
            training_months: list[int] of months (1-12) to use for threshold optimization
            testing_months: list[int] of months (1-12) to evaluate on
            metric: str, which metric to optimize ("tss", "accuracy", "iou", or "combined")

        Returns:
            best_threshold: float
            df_test: pd.DataFrame with metrics on testing months
        """
        # training_months = [str(t) for t in training_months]
        # testing_months = [str(t) for t in testing_months]

        print("Loading training predictions and ground truth...")
        predictions = []
        ground_truth = []
        for pred_file in Path(pred_folder).iterdir():
            date_str = extract_date(pred_file.name)
            if not date_str:
                continue
            month = datetime.strptime(date_str, "%Y-%m-%d").month
            if month not in training_months:
                continue

            gt_file = get_file_by_date(pred_file.name, gt_folder)
            if not gt_file:
                continue

            try:
                predictions.append(np.load(pred_file))
                ground_truth.append(np.load(gt_file))
            except Exception as e:
                print(f"Error loading {pred_file} or {gt_file}: {e}")

        if not predictions:
            raise ValueError("No training data found for given months")

        # TODO figure out if this is needed
        print("Determining candidate thresholds...")
        flattened, thresholds = StatisticsHelper.determine_thresholds(predictions)

        print(f"Optimizing threshold using netric weights = {metric_weights}")
        # print("PREDICTIONS SHAPE DELETE ", predictions.shape)
        best_threshold, best_metrics, _ = (
            StatisticsHelper.optimize_threshold_by_combined_metric(
                thresholds,
                predictions,
                ground_truth,
                weights=metric_weights,
            )
        )

        print(f"Best threshold ({metric_weights}) = {best_threshold:.4f}")
        print("Evaluating on testing months...")
        df_test = StatisticsHelper.compare_datasets(
            pred_folder, gt_folder, threshold=best_threshold, months=testing_months
        )

        return best_threshold, df_test

    @staticmethod
    def optimize_thresholds_for_metrics(thresholds, predictions, ground_truth):
        y_true = np.concatenate([gt.flatten() for gt in ground_truth])

        all_metric_outputs = []
        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            metrics = StatisticsHelper.test_single_threshold(
                threshold, predictions, y_true
            )
            all_metric_outputs.append(metrics)

        # Determine the best threshold based on the specified metric
        best_accuracy = max(all_metric_outputs, key=lambda x: x["accuracy"])
        best_tss = max(all_metric_outputs, key=lambda x: x["tss"])
        best_iou = max(all_metric_outputs, key=lambda x: x["iou"])

        return best_accuracy, best_tss, best_iou, all_metric_outputs

    @staticmethod
    def optimize_threshold_by_combined_metric(
        thresholds, predictions, ground_truth, weights=None
    ):
        """
        Find the best threshold that optimizes a weighted combination of metrics.

        Args:
            thresholds: list of threshold values to test
            predictions: list of model prediction masks
            ground_truth: list of ground truth masks
            weights: dict with weights for each metric, e.g., {'tss': 0.4, 'accuracy': 0.2, 'iou': 0.2}

        Returns:
            best_threshold: the threshold that maximizes the combined score
            best_metrics: the full metric dictionary at the best threshold
            all_metrics: list of all metrics evaluated at each threshold
        """
        if weights is None:
            # Default: equal weights TODO figure out if this NOT default setting is what was used in the tables, since it isn't even here.
            weights = {"tss": 0.3333, "accuracy": 0.3333, "iou": 0.3333}

        # Flatten all ground truth arrays
        y_true = np.concatenate([gt.flatten() for gt in ground_truth])

        all_metric_outputs = []

        for threshold in tqdm(thresholds, desc="Optimizing combined metric"):
            metrics = StatisticsHelper.test_single_threshold(
                threshold, predictions, y_true
            )
            if metrics is None:
                continue

            # Combined score: weighted sum of selected metrics
            combined_score = sum(metrics[m] * weights.get(m, 0) for m in weights)
            metrics["combined_score"] = combined_score
            all_metric_outputs.append(metrics)

        if not all_metric_outputs:
            return None, None, []

        # Find the threshold with the highest combined score
        best_metrics = max(all_metric_outputs, key=lambda x: x["combined_score"])
        best_threshold = best_metrics["threshold"]

        return best_threshold, best_metrics, all_metric_outputs

    @staticmethod
    # TODO make thsi work with training and testing months
    def compare_datasets(pred_folder, gt_folder, threshold=0.5, months=None):
        """
        Compare two datasets (prediction vs ground truth) by matching filenames via date.

        Args:
            pred_folder: Path to folder with predicted masks
            gt_folder: Path to folder with ground truth masks
            threshold: Threshold for binarizing predicted masks
            months: List of allowed months [1-12]. If None, all months are included.

        Returns:
            pd.DataFrame with per-file metrics and overall mean
        """
        pred_folder = Path(pred_folder)
        gt_folder = Path(gt_folder)

        results = []

        # Iterate over prediction files
        for pred_path in pred_folder.iterdir():
            if not pred_path.is_file():
                continue

            # Find matching GT file
            pred_file = pred_path.name
            date_str = extract_date(pred_file)
            if not date_str:
                continue

            # Filter by month if requested
            if months is not None:
                try:
                    month = datetime.strptime(date_str, "%Y-%m-%d").month
                except ValueError:
                    print(f"Could not parse date {date_str} in file {pred_file}")
                    continue

                if month not in months:
                    continue

            gt_file = get_file_by_date(pred_file, gt_folder)
            if not gt_file:
                print(f"No ground truth match found for {pred_file}")
                continue

            gt_path = Path(gt_file)

            # Load arrays
            try:
                pred = np.load(pred_path)
                gt = np.load(gt_path)
            except Exception as e:
                print(f"Error loading {pred_file} or {gt_file}: {e}")
                continue

            # Flatten & binarize
            pred_bin = (pred > threshold).astype(np.uint8).flatten()
            gt_bin = (gt > 0).astype(np.uint8).flatten()

            # Compute metrics
            metrics = StatisticsHelper.calculate_metrics(gt_bin, pred_bin)
            metrics["file"] = pred_file
            results.append(metrics)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Add row with mean metrics
        mean_metrics = df.drop(columns=["file"]).mean(numeric_only=True).to_dict()
        mean_metrics["file"] = "MEAN"
        df = pd.concat([df, pd.DataFrame([mean_metrics])], ignore_index=True)

        return df
