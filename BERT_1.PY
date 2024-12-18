from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader

# Load the tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification for IMDb

# Load the IMDb dataset
dataset = load_dataset('imdb', split='test[:10%]')  # Load 10% of the test set for evaluation

# Tokenize the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Convert the dataset to PyTorch tensors
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Create a DataLoader
dataloader = DataLoader(encoded_dataset, batch_size=8)  # Adjust batch size as needed

# Evaluate the model
all_predictions = []
all_labels = []

for batch in dataloader:
    inputs = {key: batch[key].to(model.device) for key in ['input_ids', 'attention_mask']}
    labels = batch['label'].to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Compute metrics
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
accuracy = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item() / len(all_labels)
cohen_kappa = cohen_kappa_score(all_labels, all_predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Cohen's Kappa: {cohen_kappa:.4f}")
