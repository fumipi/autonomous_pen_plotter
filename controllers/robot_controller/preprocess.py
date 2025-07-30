import cv2
import numpy as np
import math
from PIL import Image

def convert_to_panorama(image):
    """通常のカメラ画像をパノラマ形式に変換"""
    height, width = image.shape[:2]
    
    # カメラの視野角を仮定（実際のカメラパラメータに合わせて調整）
    fov_h = 60  # 水平視野角（度）
    fov_v = 45  # 垂直視野角（度）
    
    # パノラマ画像のサイズ
    pano_width = 512
    pano_height = 256
    
    # 球面座標への変換
    pano = np.zeros((pano_height, pano_width, 3), dtype=np.uint8)
    
    for y in range(pano_height):
        for x in range(pano_width):
            # パノラマ座標を球面座標に変換
            theta = (x / pano_width) * 2 * math.pi - math.pi  # -π to π
            phi = (y / pano_height) * math.pi  # 0 to π
            
            # 球面座標をカメラ座標に変換
            # カメラは上向き（Z軸正方向）を仮定
            x_cam = math.sin(phi) * math.cos(theta)
            y_cam = math.sin(phi) * math.sin(theta)
            z_cam = math.cos(phi)
            
            # カメラ座標を画像座標に変換
            if z_cam > 0:  # カメラの前方
                u = int((x_cam / z_cam) * width / (2 * math.tan(math.radians(fov_h/2))) + width/2)
                v = int((y_cam / z_cam) * height / (2 * math.tan(math.radians(fov_v/2))) + height/2)
                
                # 画像範囲内かチェック
                if 0 <= u < width and 0 <= v < height:
                    pano[y, x] = image[v, u]
    
    return pano

def align_horizon(image):
    """水平線を基準に画像を整列"""
    # グレースケール変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)
    
    # 直線検出
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        # 水平線の角度を計算
        horizontal_angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # 水平線の範囲（0度または180度に近い）
            if angle < 10 or angle > 170:
                horizontal_angles.append(angle)
        
        if horizontal_angles:
            # 平均角度を計算
            avg_angle = np.mean(horizontal_angles)
            if avg_angle > 90:
                avg_angle -= 180
            
            # 画像を回転
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            aligned_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            return aligned_image
    
    return image

def preprocess_for_horizonnet(image):
    """HorizonNet用の完全な前処理パイプライン"""
    # 1. パノラマ変換
    pano = convert_to_panorama(image)
    
    # 2. 水平線整列
    aligned = align_horizon(pano)
    
    # 3. リサイズ
    resized = cv2.resize(aligned, (512, 256))
    
    return resized

def save_debug_image(image, filename):
    """デバッグ用に画像を保存"""
    cv2.imwrite(filename, image)

if __name__ == "__main__":
    # テスト用
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    processed = preprocess_for_horizonnet(test_image)
    save_debug_image(processed, "debug_processed.png")
