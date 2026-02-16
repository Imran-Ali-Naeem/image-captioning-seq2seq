# 🖼️ Neural Storyteller - Image Captioning with Seq2Seq

Generate natural language descriptions for images using Sequence-to-Sequence architecture with LSTM and ResNet50.

## 🚀 Live Demo

Try it live: **[HuggingFace Space](https://huggingface.co/spaces/ImranAliNaeem/image-captioning-seq2seq)**



## 📊 Quick Results

- **BLEU-4:** 0.1224
- **Precision:** 0.3828 | **Recall:** 0.2388 | **F1:** 0.2866
- **Training:** 28 epochs, early stopping at validation loss 3.08

## 🏗️ Architecture
```
ResNet50 (2048-dim) → Encoder (512-dim) → LSTM Decoder → Caption
```

**Details:**
- Feature extraction: Pre-trained ResNet50
- Vocabulary: 19,774 words from Flickr30k
- Model: Encoder-Decoder with LSTM + Dropout(0.5)
- Training: Adam optimizer (lr=2e-4), gradient clipping, early stopping

## 📁 Repository Contents

- `app.py` - Gradio web interface
- `best_model.pth` - Trained model weights (71.3 MB)
- `word2idx.pkl` / `idx2word.pkl` - Vocabulary mappings
- `notebook/` - Full training notebook
- `images/` - Sample results and visualizations

## 🛠️ Local Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/image-captioning-seq2seq.git
cd image-captioning-seq2seq

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py
```

## 📊 Sample Outputs


<img width="1308" height="559" alt="result-1" src="https://github.com/user-attachments/assets/c23069f3-760a-4efe-9e48-704cbfb43360" />

<img width="1291" height="586" alt="result-2" src="https://github.com/user-attachments/assets/eae6d62e-d80b-44c9-8958-6a6d5fe20df1" />



![Training Curve](images/loss_curve.png)

## 🔧 Tech Stack

`PyTorch` • `ResNet50` • `LSTM` • `Gradio` • `HuggingFace` • `Flickr30k`

## 📚 Dataset

Flickr30k: 31,783 images, 158,915 captions
[Download from Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr30k)

## 📄 License

MIT License - see [LICENSE](LICENSE)

## 📧 Contact

**Your Name**
- LinkedIn: [Profile](https://www.linkedin.com/posts/imran-ali-naeem_genai-deeplearning-pytorch-activity-7428233465952772096-ipMO
)
- HuggingFace: [Space](https://huggingface.co/spaces/ImranAliNaeem/image-captioning-seq2seq)

---

⭐ Star this repo if you found it helpful!
