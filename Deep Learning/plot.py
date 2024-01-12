import pickle
import matplotlib.pyplot as plt
import os

# Load the training history from the saved file
with open("history.pkl", "rb") as f:
    history = pickle.load(f)

# Create a directory for storing plots
plots_dir = "Plots"
os.makedirs(plots_dir, exist_ok=True)

# Plot training and validation accuracy
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy Over Epochs')
plt.savefig(os.path.join(plots_dir, 'accuracy_plot.png'), dpi=300)
plt.show()

# Plot training and validation loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.savefig(os.path.join(plots_dir, 'loss_plot.png'), dpi=300)
plt.show()

# Final comparison plot
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.title('Final Training and Validation Comparison')
plt.savefig(os.path.join(plots_dir, 'final_comparison_plot.png'), dpi=300)
plt.show()

print("Plots saved in the 'Plots' directory.")


# Plot training and validation loss
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.savefig('plots/training_validation_loss.png', dpi=300)
plt.show()


