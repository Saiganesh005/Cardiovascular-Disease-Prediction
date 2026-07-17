import logging
import os

# Create logs directory if it doesn't exist
log_dir = "output/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging to write to a file and the console
log_file_path = os.path.join(log_dir, "model_training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'), # Overwrite the log file for each run
        logging.StreamHandler() # Output to console
    ]
)

logger = logging.getLogger(__name__)

logger.info("--- Model Training Results Summary ---")

# Extract traditional model accuracies from the 'scores' DataFrame
model_results = {}
for index, row in scores.iterrows():
    model_results[row["Model"]] = row["Accuracy"]

# Add ANN accuracy (observed from previous cell's output)
# In a more robust setup, this would be retrieved programmatically from a saved variable or file
ann_accuracy = 0.7379 # From output of cell 5Rill-4nKbHA
model_results["ANN"] = ann_accuracy

# Log individual model accuracies
for model_name, accuracy in model_results.items():
    logger.info(f"{model_name} Accuracy: {accuracy:.4f}")

# Determine and log the overall best performing model
best_model_name = max(model_results, key=model_results.get)
best_accuracy = model_results[best_model_name]
logger.info(f"Overall Best Performing Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

logger.info(f"All log outputs have been saved to: {log_file_path}")

# Also print a summary to the console for immediate user feedback
print("\n--- Summary of Model Accuracies ---")
for model_name, accuracy in model_results.items():
    print(f"{model_name}: {accuracy:.4f}")
print(f"Overall Best Performing Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("--- End Summary ---")
