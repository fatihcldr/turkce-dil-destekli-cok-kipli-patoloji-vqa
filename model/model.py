import torch
import torch.nn as nn
from transformers import ViTModel, BertModel


class PathVQAModel(nn.Module):
    def __init__(self, num_classes=2, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False

        vit_dim = self.vit.config.hidden_size    # 768
        bert_dim = self.bert.config.hidden_size  # 768

        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        image_feat = self.vit(pixel_values=pixel_values).pooler_output
        text_feat = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        fused = torch.cat([image_feat, text_feat], dim=-1)
        return self.fusion(fused)
