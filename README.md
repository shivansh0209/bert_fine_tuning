# Twitter Sentiment Classifier: Fine-Tuning & Inference Optimization

A production-grade NLP pipeline that fine-tunes a `BERT-base` model on a Twitter dataset for 3-class sentiment analysis (**Negative**, **Neutral**, **Positive**). This project highlights end-to-end engineering practices including dynamic padding, custom tokenization mapping, and advanced inference optimizations.

---

## 🚀 Key Features & Architectural Highlights

* **Dynamic Batch Padding:** Utilizes `DataCollatorWithPadding` to dynamically pad sequences to the maximum length per batch rather than the entire dataset, saving significant GPU memory and reducing compute bottlenecks.
* **Inference Optimization:** Implements `bitsandbytes` 8-bit quantization and `device_map="auto"` for low-latency, memory-efficient deployment configurations.

---

## 📊 Evaluation & Performance Metrics

The model reached optimal generalized performance at **Epoch 1**, achieving **~80% accuracy** (on par with baseline human annotator agreement for noisy, informal tweet data).

### Training Progress Logs

| Epoch | Training Loss | Validation Loss | Accuracy | F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.5479 | 0.5124 | **79.95%** | **79.97%** |
| **2** | 0.4209 | 0.5331 | 79.24% | 79.13% |

> 💡 **Note on Overfitting:** At Epoch 2, the training loss dropped significantly while validation loss began to climb, signaling the start of overfitting. The pipeline automatically utilized `load_best_model_at_end=True` to roll back and permanently preserve the optimal Epoch 1 checkpoint.