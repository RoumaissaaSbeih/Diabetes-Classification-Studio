from diabetes_dashboard.modeling import train_and_save_artifacts


if __name__ == "__main__":
    metrics = train_and_save_artifacts(force=True)
    final_model = metrics["final_model"]
    test_metrics = final_model["test_metrics"]
    print("Training complete.")
    print(f'Model: {final_model["name"]}')
    print(f'Best parameters: {final_model["best_params"]}')
    print(
        "Test metrics -> "
        f'accuracy={test_metrics["accuracy"]:.3f}, '
        f'f1={test_metrics["f1"]:.3f}, '
        f'roc_auc={test_metrics["roc_auc"]:.3f}'
    )
