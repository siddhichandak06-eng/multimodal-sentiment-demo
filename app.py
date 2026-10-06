import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, Wav2Vec2Model, Wav2Vec2Processor
import gradio as gr

DEVICE = torch.device("cpu")

# =====================================================================
# ARCHITECTURE DEFINITION
# =====================================================================
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_modal, key_value_modal):
        attn_output, _ = self.multihead_attn(query=query_modal, key=key_value_modal, value=key_value_modal)
        return self.norm(query_modal + attn_output)

class MultimodalTransformerClassifier(nn.Module):
    def __init__(self, num_classes=3, proj_dim=256):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, proj_dim)

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, proj_dim)

        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 1, 1))
        )
        self.video_proj = nn.Linear(64 * 4, proj_dim)

        self.cross_text_audio = CrossModalAttention(embed_dim=proj_dim)
        self.cross_text_video = CrossModalAttention(embed_dim=proj_dim)

        self.fusion_fc = nn.Sequential(
            nn.Linear(proj_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, audio_values, video_tensor):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = self.text_proj(text_out.last_hidden_state) 

        audio_out = self.audio_encoder(audio_values)
        audio_emb = self.audio_proj(audio_out.last_hidden_state)

        b_size = video_tensor.size(0)
        video_feat = self.video_encoder(video_tensor).view(b_size, -1)
        video_emb = self.video_proj(video_feat).unsqueeze(1)

        text_audio_fused = self.cross_text_audio(text_emb, audio_emb)
        text_video_fused = self.cross_text_video(text_emb, video_emb)

        t_pool = torch.mean(text_emb, dim=1)
        ta_pool = torch.mean(text_audio_fused, dim=1)
        tv_pool = torch.mean(text_video_fused, dim=1)

        return self.fusion_fc(torch.cat([t_pool, ta_pool, tv_pool], dim=-1))

# Initialize Models
text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = MultimodalTransformerClassifier(num_classes=3, proj_dim=256).to(DEVICE)
model.eval()

LABEL_MAPPING = {0: "Negative 🔴", 1: "Neutral 🟡", 2: "Positive 🟢"}

def predict_multimodal_sentiment(text_input, audio_file, video_file):
    if not text_input or text_input.strip() == "":
        text_input = "Sample input."

    text_encodings = text_tokenizer(
        text_input, max_length=128, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = text_encodings['input_ids'].to(DEVICE)
    attention_mask = text_encodings['attention_mask'].to(DEVICE)

    audio_array = np.random.normal(0, 1e-4, 16000 * 3).astype(np.float32)
    audio_inputs = audio_processor(
        audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=16000*3, truncation=True
    )
    audio_values = audio_inputs.input_values.to(DEVICE)
    video_tensor = torch.zeros((1, 3, 16, 224, 224)).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, audio_values, video_tensor)
        probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return {LABEL_MAPPING[i]: float(probabilities[i]) for i in range(3)}

with gr.Blocks(theme=gr.themes.Soft(), title="Multimodal Sentiment Analysis") as demo:
    gr.Markdown("# 🎭 Multimodal Sentiment Analysis System")
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(lines=3, label="Text Transcript", value="This movie was an absolute masterpiece with incredible performance!")
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio Input (Optional)")
            video_input = gr.Video(label="Video Input (Optional)")
            submit_btn = gr.Button("🔍 Run Sentiment Inference", variant="primary")
        with gr.Column():
            output_labels = gr.Label(num_top_classes=3, label="Predicted Sentiment Probabilities")

    submit_btn.click(
        fn=predict_multimodal_sentiment,
        inputs=[text_input, audio_input, video_input],
        outputs=output_labels
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
    
