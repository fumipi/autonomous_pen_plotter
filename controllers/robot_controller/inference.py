import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json
import math
from PIL import Image
import os

class HorizonNet(nn.Module):
    def __init__(self, backbone='resnet50', no_rnn=False):
        super(HorizonNet, self).__init__()
        
        # Backbone network
        if backbone == 'resnet50':
            from torchvision.models import resnet50
            backbone = resnet50(pretrained=False)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            backbone_dim = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
        # RNN layers
        if not no_rnn:
            self.rnn = nn.LSTM(backbone_dim, 512, num_layers=2, batch_first=True, dropout=0.5)
            rnn_dim = 512
        else:
            self.rnn = None
            rnn_dim = backbone_dim
        
        # Output layers
        self.fc1 = nn.Linear(rnn_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Final output layers for different tasks
        self.fc_cor = nn.Linear(128, 1)  # corner prediction
        self.fc_bon = nn.Linear(128, 1)  # boundary prediction
        
    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process through backbone
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)  # [B*seq_len, 2048, H', W']
        
        # Global average pooling
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # [B*seq_len, 2048]
        features = features.view(batch_size, seq_len, -1)  # [B, seq_len, 2048]
        
        # RNN processing
        if self.rnn is not None:
            features, _ = self.rnn(features)  # [B, seq_len, 512]
        
        # Final features
        features = features[:, -1, :]  # Take last timestep
        
        # FC layers
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Outputs
        cor = self.fc_cor(x)  # Corner prediction
        bon = self.fc_bon(x)  # Boundary prediction
        
        return cor, bon

class LayoutEstimator:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = HorizonNet(backbone='resnet50', no_rnn=False)
        
        # Load pretrained weights
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            # Create a dummy model for testing
            self.model = None
        
        # Load room map for reference
        try:
            with open('room_map.json', 'r') as f:
                self.room_map = json.load(f)
        except FileNotFoundError:
            print("room_map.json not found, using default room layout")
            self.room_map = {"walls": []}
    
    def preprocess_image(self, image):
        """画像をHorizonNet用に前処理"""
        if self.model is None:
            return None
            
        # Resize to standard size
        image = cv2.resize(image, (512, 256))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return image.to(self.device)
    
    def estimate_layout(self, image):
        """画像からレイアウトを推定"""
        if self.model is None:
            return 0.0, 0.0  # Dummy values
            
        with torch.no_grad():
            # Preprocess image
            input_tensor = self.preprocess_image(image)
            if input_tensor is None:
                return 0.0, 0.0
            
            # Forward pass
            cor, bon = self.model(input_tensor)
            
            # Convert predictions to numpy
            cor = cor.cpu().numpy()[0, 0]
            bon = bon.cpu().numpy()[0, 0]
            
            return cor, bon
    
    def estimate_position_orientation(self, image, compass_heading):
        """画像とコンパス情報から自己位置・向きを推定"""
        # Layout estimation
        cor, bon = self.estimate_layout(image)
        
        # Simple heuristic for position estimation based on room layout
        # This is a simplified version - in practice, you'd want more sophisticated matching
        
        # Use room map to estimate position
        estimated_x, estimated_y = self._estimate_position_from_layout(cor, bon)
        estimated_heading = compass_heading
        
        # 推定結果の信頼性をチェック
        confidence = self._check_estimation_confidence(cor, bon)
        
        # 信頼性が低い場合は推定を拒否
        if confidence < 0.3:  # 30%未満の信頼性
            print(f"Low confidence estimation ({confidence:.2f}), rejecting position update")
            return None, None, None
        
        print(f"Position estimation confidence: {confidence:.2f}")
        return estimated_x, estimated_y, estimated_heading
    
    def _estimate_position_from_layout(self, cor, bon):
        """レイアウト情報から位置を推定（改善版）"""
        # 部屋の境界を取得
        walls = self.room_map.get("walls", [])
        if not walls:
            # デフォルトの部屋サイズ
            room_width = 10.0
            room_height = 10.0
            room_center_x = 0.0
            room_center_y = 0.0
        else:
            # 壁の座標から部屋の境界を計算
            x_coords = []
            y_coords = []
            for wall in walls:
                for point in wall:
                    x_coords.append(point[0])
                    y_coords.append(point[1])
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            room_width = max_x - min_x
            room_height = max_y - min_y
            room_center_x = (min_x + max_x) / 2.0
            room_center_y = (min_y + max_y) / 2.0
        
        # レイアウト特徴量を正規化
        # cor, bon は -1 から 1 の範囲と仮定
        cor_normalized = np.clip(cor, -1.0, 1.0)
        bon_normalized = np.clip(bon, -1.0, 1.0)
        
        # 部屋の中心からの相対位置を計算
        # より小さな調整係数を使用して、急激な位置変化を防ぐ
        offset_x = cor_normalized * room_width * 0.05  # 5%の調整
        offset_y = bon_normalized * room_height * 0.05
        
        # 推定位置を計算
        estimated_x = room_center_x + offset_x
        estimated_y = room_center_y + offset_y
        
        # 部屋の境界内に制限
        if walls:
            estimated_x = np.clip(estimated_x, min_x + 0.5, max_x - 0.5)
            estimated_y = np.clip(estimated_y, min_y + 0.5, max_y - 0.5)
        
        return estimated_x, estimated_y

    def _check_estimation_confidence(self, cor, bon):
        """推定結果の信頼性をチェック"""
        # レイアウト特徴量の絶対値が大きいほど信頼性が高い
        cor_confidence = min(abs(cor), 1.0)
        bon_confidence = min(abs(bon), 1.0)
        
        # 平均信頼性を計算
        confidence = (cor_confidence + bon_confidence) / 2.0
        
        return confidence

def main():
    # Example usage
    model_path = 'ckpt/resnet50_rnn__mp3d.pth'
    estimator = LayoutEstimator(model_path)
    
    # Load a test image (you would get this from the camera)
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Estimate position and orientation
    x, y, heading = estimator.estimate_position_orientation(test_image, 0.0)
    print(f"Estimated position: ({x:.3f}, {y:.3f}), heading: {heading:.3f}")

if __name__ == "__main__":
    main()
