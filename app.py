import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

# Force CPU
device = torch.device("cpu")

# Load vocabulary
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)
START_TOKEN = "<start>"
END_TOKEN = "<end>"

# Model architecture
class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU()
    
    def forward(self, features):
        return self.relu(self.fc(features))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, captions, hidden):
        embeddings = self.dropout(self.embedding(captions))
        outputs, hidden = self.lstm(embeddings, hidden)
        outputs = self.fc(self.dropout(outputs))
        return outputs, hidden

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, feature_dim=2048, embed_dim=256, hidden_dim=512):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim)
    
    def forward(self, features, captions):
        encoded = self.encoder(features)
        h0 = encoded.unsqueeze(0)
        c0 = torch.zeros_like(h0)
        outputs, _ = self.decoder(captions, (h0, c0))
        return outputs

# Load caption model
print("Loading caption model...")
model = ImageCaptioningModel(vocab_size=vocab_size)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Load ResNet (FIXED)
print("Loading ResNet50...")
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

print("✓ Models loaded successfully!")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Generate caption
def generate_caption(image):
    try:
        img = Image.fromarray(image).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            # Extract features
            feature = resnet(img_tensor).view(1, -1)
            
            # Generate caption
            encoded = model.encoder(feature)
            h0 = encoded.unsqueeze(0)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
            
            input_word = torch.tensor([[word2idx[START_TOKEN]]])
            caption = []
            
            for _ in range(30):
                output, hidden = model.decoder(input_word, hidden)
                predicted_idx = output.argmax(dim=2).item()
                
                if predicted_idx == word2idx[END_TOKEN]:
                    break
                
                caption.append(idx2word[predicted_idx])
                input_word = torch.tensor([[predicted_idx]])
        
        return ' '.join(caption) if caption else "Could not generate caption"
    
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Generated Caption"),
    title="🖼️ Image Captioning - Neural Storyteller",
    description="Upload an image to generate a caption using Seq2Seq LSTM model",
    examples=None
)

iface.launch()
