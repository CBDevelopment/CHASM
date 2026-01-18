#from run_stats.paper.metrics import compare_folders
#from src.polished_pipeline.datasets.combined_dataset import *
#from train_model import preprocess_chronnos
from stats_updated.stats_helper import StatisticsHelper 
from datasets.folder_utils import *
import pandas as pd

if __name__=="__main__":
    pred_folders = [
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1407/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM967/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/Finetuned/CHASM1111_finetuned_1e-5/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/Finetuned/CHASM1111_finetuned_1e-6/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/Finetuned/CHASM1407_finetuned_1e-5/predictions",
        r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/Finetuned/CHASM1407_finetuned_1e-6/predictions"
    ]
    truth_folder = r"/home/egsmith/scratch/CoronalHoles/src/polished_pipeline/PREDICTIONS/CHASM1111/chronnos/inputs/mask/512"
    fixed_threshold = 0.5
    metrics_to_optimize = [({'tss': 0.3333, 'accuracy': 0.3333, 'iou': 0.3333}, "balanced"),
                           ({'tss': 1, 'accuracy': 0, 'iou': 0}, "tss"),
                           ({'tss': 0, 'accuracy': 1, 'iou': 0}, "accuracy"),
                           ({'tss': 0, 'accuracy': 0, 'iou': 1}, "iou")]
    results = []

    for pred_folder in pred_folders:
        print(f"Generating table for folder {pred_folder}")

        name = Path(pred_folder).parent.name  # e.g., CHASM1111

        # --- Fixed threshold run ---
        df_fixed = StatisticsHelper.compare_datasets(pred_folder, truth_folder, threshold=fixed_threshold, months=[11,12])
        mean_fixed = df_fixed[df_fixed["file"] == "MEAN"].iloc[0]
        results.append({
            "name": name,
            "method": "fixed",
            "threshold": fixed_threshold,
            "accuracy": mean_fixed["accuracy"],
            "tss": mean_fixed["tss"],
            "iou": mean_fixed["iou"],
            "dice": mean_fixed.get("dice", None)
        })

        # TODO delete this section
        # --- Optimized threshold run ---
        # best_threshold, df_opt = StatisticsHelper.optimize_and_evaluate(pred_folder, truth_folder)
        # optimized = True

        # mean_opt = df_opt[df_opt["file"] == "MEAN"].iloc[0]
        # results.append({
        #     "name": name,
        #     "method": "optimized" if optimized else "fixed",
        #     "threshold": best_threshold,
        #     "accuracy": mean_opt["accuracy"],
        #     "tss": mean_opt["tss"],
        #     "iou": mean_opt["iou"],
        #     "dice": mean_opt.get("dice", None)
        # })

        # --- Optimized thresholds for each metric ---
        for metric_dict, method_name in metrics_to_optimize:
            print(f"Optimizing metric {metric_dict}")
            best_threshold, df_opt = StatisticsHelper.optimize_and_evaluate(
                pred_folder, truth_folder, metric_weights=metric_dict,
                training_months=[11, 12], testing_months=[11, 12], # TODO check if we can optimize thresholds just on TESTING or if training needed too
            )
            mean_opt = df_opt[df_opt["file"] == "MEAN"].iloc[0]
            results.append({
                "name": name,
                "method": "optimized",
                "optimized_metric": method_name,
                "threshold": best_threshold,
                "accuracy": mean_opt["accuracy"],
                "tss": mean_opt["tss"],
                "iou": mean_opt["iou"],
                "dice": mean_opt.get("dice", None)
            })

    # --- Convert to DataFrame ---
    results_df = pd.DataFrame(results)

    # Save CSV
    results_df.to_csv("threshold_comparison_results.csv", index=False)

    print("Results DataFrame:")
    print(results_df)

    # LaTeX table
    latex_table = results_df.to_latex(index=False, float_format="%.4f")
    print("\nLaTeX Table:\n")
    print(latex_table)