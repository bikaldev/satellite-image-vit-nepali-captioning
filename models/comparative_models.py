import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Optional, Tuple
import os

# ==========================================
# 1. Generic CNN Classifier (ResNet50 / VGG16)
# ==========================================

class CNNClassifier(nn.Module):
    def __init__(self, model_type: str = 'resnet50', num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == 'resnet50':
            self.backbone = models.resnet50(weights='DEFAULT' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_type == 'vgg16':
            self.backbone = models.vgg16(weights='DEFAULT' if pretrained else None)
            num_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[6] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def predict(self, x: torch.Tensor, return_probs: bool = True) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            return {'logits': logits, 'probs': probs, 'predictions': preds}

    def save_checkpoint(self, path, epoch, metrics=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
            'num_classes': self.num_classes,
            'metrics': metrics
        }, path)

# ==========================================
# 2. CNN + LSTM Captioner (with Attention)
# ==========================================

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha

class CNNLSTMCaptioner(nn.Module):
    def __init__(self, model_type, tokenizer_name, embed_size=256, hidden_size=512, device='cuda'):
        super().__init__()
        self.device = device
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab_size = len(self.tokenizer)
        
        if model_type == 'resnet50':
            resnet = models.resnet50(weights='DEFAULT')
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.encoder_dim = 2048
        elif model_type == 'vgg16':
            vgg = models.vgg16(weights='DEFAULT')
            self.encoder = vgg.features
            self.encoder_dim = 512
        else:
            raise ValueError("Unsupported model type")
            
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(self.encoder_dim, hidden_size, 256)
        self.lstm = nn.LSTMCell(embed_size + self.encoder_dim, hidden_size, bias=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_h = nn.Linear(self.encoder_dim, hidden_size)
        self.init_c = nn.Linear(self.encoder_dim, hidden_size)

    def forward(self, images, captions):
        batch_size = images.size(0)
        features = self.encoder(images) # [batch, dim, 7, 7]
        features = features.permute(0, 2, 3, 1).view(batch_size, -1, self.encoder_dim) # [batch, 49, dim]
        
        embeddings = self.embed(captions)
        
        avg_features = features.mean(dim=1)
        h = self.init_h(avg_features)
        c = self.init_c(avg_features)
        
        max_len = captions.size(1)
        outputs = torch.zeros(batch_size, max_len, len(self.tokenizer)).to(self.device)
        
        for t in range(max_len):
            context, _ = self.attention(features, h)
            h, c = self.lstm(torch.cat([embeddings[:, t, :], context], dim=1), (h, c))
            outputs[:, t, :] = self.fc(h)
            
        return outputs

    def generate_caption(self, images, max_length=50):
        self.eval()
        batch_size = images.size(0)
        features = self.encoder(images).permute(0, 2, 3, 1).view(batch_size, -1, self.encoder_dim)
        
        avg_features = features.mean(dim=1)
        h = self.init_h(avg_features)
        c = self.init_c(avg_features)
        
        start_token = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
        current_token = torch.full((batch_size,), start_token, dtype=torch.long).to(self.device)
        
        generated_tokens = []
        for _ in range(max_length):
            embedding = self.embed(current_token)
            context, _ = self.attention(features, h)
            h, c = self.lstm(torch.cat([embedding, context], dim=1), (h, c))
            logits = self.fc(h)
            current_token = torch.argmax(logits, dim=1)
            generated_tokens.append(current_token.unsqueeze(1))
            
        generated_tokens = torch.cat(generated_tokens, dim=1)
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def save_checkpoint(self, path, epoch, metrics=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_type': self.model_type,
            'tokenizer_name': self.tokenizer.name_or_path,
            'metrics': metrics
        }, os.path.join(path, "checkpoint.pt"))

# ==========================================
# 3. ResNet50 + mGPT Captioner
# ==========================================

class ResNetmGPTCaptioner(nn.Module):
    def __init__(self, tokenizer_name, decoder_model_name="ai-forever/mGPT", device='cuda'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        resnet = models.resnet50(weights='DEFAULT')
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.encoder_dim = 2048
        
        # Load mGPT config and set add_cross_attention=True
        config = AutoConfig.from_pretrained(decoder_model_name)
        config.add_cross_attention = True
        config.is_decoder = True
        
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, config=config)
        
        # Projection layer to match mGPT dimension (usually 768 or 1024)
        self.proj = nn.Linear(self.encoder_dim, self.decoder.config.hidden_size)

    def forward(self, images, captions):
        features = self.encoder(images) # [batch, 2048, 7, 7]
        features = features.view(features.size(0), self.encoder_dim, -1).permute(0, 2, 1) # [batch, 49, 2048]
        encoder_hidden_states = self.proj(features)
        
        outputs = self.decoder(input_ids=captions, encoder_hidden_states=encoder_hidden_states)
        return outputs.logits

    def generate_caption(self, images, max_length=50):
        self.eval()
        batch_size = images.size(0)
        features = self.encoder(images).view(batch_size, self.encoder_dim, -1).permute(0, 2, 1)
        encoder_hidden_states = self.proj(features)
        
        # Start token
        idx = torch.full((batch_size, 1), self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id, dtype=torch.long).to(self.device)
        
        for _ in range(max_length):
            outputs = self.decoder(input_ids=idx, encoder_hidden_states=encoder_hidden_states)
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)
            if (next_token == self.tokenizer.eos_token_id).all():
                break
                
        return self.tokenizer.batch_decode(idx[:, 1:], skip_special_tokens=True)

    def save_checkpoint(self, path, epoch, metrics=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'tokenizer_name': self.tokenizer.name_or_path,
            'metrics': metrics
        }, os.path.join(path, "checkpoint.pt"))
