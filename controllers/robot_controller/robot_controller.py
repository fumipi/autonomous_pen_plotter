from controller import Robot
import math
import time
import numpy as np
import cv2
from inference import LayoutEstimator
from preprocess import preprocess_for_horizonnet

# --- シミュレーション設定 ---
TIMESTEP      = 32               # ms
DT            = TIMESTEP / 1000  # s

# --- ロボット幾何パラメータ ---
WHEEL_DIAMETER = 0.05  # m
WHEEL_BASE     = 0.09  # m
RADIUS         = WHEEL_DIAMETER / 2.0

# --- 自己位置推定設定 ---
LAYOUT_UPDATE_INTERVAL = 30.0  # 30秒ごとにレイアウト推定で位置リセット
INITIAL_POSITION = (0.5, 0.5)  # 初期位置（原点から少し離れた場所）

class IMUCameraCompassController(Robot):
    def __init__(self):
        super().__init__()
        
        # モーター／エンコーダー
        self.left_motor  = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_enc  = self.getDevice('left wheel sensor')
        self.right_enc = self.getDevice('right wheel sensor')
        self.left_enc.enable(TIMESTEP)
        self.right_enc.enable(TIMESTEP)
        
        # 無限回転モード
        for m in (self.left_motor, self.right_motor):
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
        
        # IMU （InertialUnit）
        self.imu = self.getDevice('inertial unit')
        self.imu.enable(TIMESTEP)
        
        # コンパス
        self.compass = self.getDevice('compass')
        self.compass.enable(TIMESTEP)
        
        # カメラ
        self.camera = self.getDevice('camera')
        self.camera.enable(TIMESTEP)
        
        # ペン
        self.pen = self.getDevice('pen')
        
        # オドメトリ初期化
        self.x = INITIAL_POSITION[0]
        self.y = INITIAL_POSITION[1]
        
        # 初回ステップでエンコーダー同期
        super().step(TIMESTEP)
        self.prev_l = self.left_enc.getValue()
        self.prev_r = self.right_enc.getValue()
        
        # HorizonNet初期化
        try:
            self.layout_estimator = LayoutEstimator('ckpt/resnet50_rnn__mp3d.pth')
            self.layout_available = True
            print("HorizonNet initialized successfully")
        except Exception as e:
            print(f"Failed to initialize HorizonNet: {e}")
            self.layout_available = False
        
        # 時間管理
        self.last_layout_update = time.time()
        self.start_time = time.time()
        
        # 目標位置
        self.target_x = 0.0
        self.target_y = 0.0
        
        # 状態管理
        self.state = "INITIAL_ESTIMATION"  # INITIAL_ESTIMATION, MOVING_TO_ORIGIN, FOLLOWING_PATH, LAYOUT_UPDATE
        
        # 経路追従の進行状況を記録
        self.current_path_index = 0
        self.current_point_index = 0
        self.paths = []
        
        print(f"Robot initialized at position: ({self.x:.3f}, {self.y:.3f})")

    def step_and_update(self):
        super().step(TIMESTEP)
        
        # エンコーダー差分で平面オドメトリ更新
        l = self.left_enc.getValue()
        r = self.right_enc.getValue()
        dL = (l - self.prev_l) * RADIUS
        dR = (r - self.prev_r) * RADIUS
        self.prev_l, self.prev_r = l, r
        dC = (dL + dR) / 2.0
        
        # θ は IMU で取るのでここでは移動だけ
        yaw = self.get_yaw()
        self.x += dC * math.cos(yaw)
        self.y += dC * math.sin(yaw)

    def get_yaw(self):
        """InertialUnit からヨー角（rad）を取得"""
        return self.imu.getRollPitchYaw()[2]

    def get_compass_heading(self):
        """コンパスから方位角（rad）を取得"""
        compass_values = self.compass.getValues()
        # コンパスは通常 [x, y, z] の値を返す
        heading = math.atan2(compass_values[1], compass_values[0])
        return heading

    def get_camera_image(self):
        """カメラから画像を取得"""
        image = self.camera.getImage()
        if image:
            # Webotsの画像形式をOpenCV形式に変換
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            
            # RGB形式で取得
            image = np.frombuffer(image, dtype=np.uint8).reshape(height, width, 4)
            # RGBAからBGRに変換
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            return image
        return None

    def estimate_position_with_layout(self):
        """HorizonNetを使って自己位置・向きを推定"""
        if not self.layout_available:
            return None, None, None
        
        try:
            # カメラ画像を取得
            image = self.get_camera_image()
            if image is None:
                return None, None, None
            
            # 画像を前処理
            processed_image = preprocess_for_horizonnet(image)
            
            # コンパス情報を取得
            compass_heading = self.get_compass_heading()
            
            # レイアウト推定
            estimated_x, estimated_y, estimated_heading = self.layout_estimator.estimate_position_orientation(
                processed_image, compass_heading
            )
            
            print(f"Layout estimation: ({estimated_x:.3f}, {estimated_y:.3f}), heading: {estimated_heading:.3f}")
            
            return estimated_x, estimated_y, estimated_heading
            
        except Exception as e:
            print(f"Layout estimation failed: {e}")
            return None, None, None

    def reset_position_with_layout(self):
        """レイアウト推定で位置をリセット"""
        estimated_x, estimated_y, estimated_heading = self.estimate_position_with_layout()
        
        if estimated_x is not None and estimated_y is not None:
            # 現在位置と推定位置の距離をチェック
            distance = math.hypot(estimated_x - self.x, estimated_y - self.y)
            
            # 急激な位置変化を防ぐ（最大2メートルまで）
            if distance > 2.0:
                print(f"Position change too large ({distance:.2f}m), rejecting update")
                return False
            
            # 位置を更新
            self.x = estimated_x
            self.y = estimated_y
            
            # 向きの補正（IMUとコンパスの組み合わせ）
            compass_heading = self.get_compass_heading()
            self.imu_offset = compass_heading - estimated_heading
            
            print(f"Position reset to: ({self.x:.3f}, {self.y:.3f})")
            return True
        else:
            print("Failed to reset position with layout estimation")
            return False

    def set_wheel_speeds(self, vl, vr):
        self.left_motor.setVelocity(vl)
        self.right_motor.setVelocity(vr)

    def stop(self):
        self.set_wheel_speeds(0.0, 0.0)

    def turn(self, rad, k_ang=6.0, tol=0.005, vmax=15.0):
        """IMU のヨー角を用いたクローズドループ回転"""
        start_yaw = self.get_yaw()
        target = start_yaw + rad
        while True:
            current = self.get_yaw()
            err = (target - current + math.pi) % (2*math.pi) - math.pi
            if abs(err) < tol:
                break
            omega = k_ang * err
            # 車輪速度へ変換
            v_r =  omega * (WHEEL_BASE/2) / RADIUS
            v_l = -omega * (WHEEL_BASE/2) / RADIUS
            # 上限クリップ
            self.set_wheel_speeds(
                max(min(v_l, vmax), -vmax),
                max(min(v_r, vmax), -vmax)
            )
            self.step_and_update()
        self.stop()

    def forward(self, dist, k_lin=20.0, tol=0.001, vmax=0.375):
        """エンコーダベースの直進クローズドループ"""
        sx, sy = self.x, self.y
        while True:
            traveled = math.hypot(self.x - sx, self.y - sy)
            err = dist - traveled
            if err < tol:
                break
            v = max(min(k_lin * err, vmax), -vmax)
            w = v / RADIUS
            self.set_wheel_speeds(w, w)
            self.step_and_update()
        self.stop()

    def goto(self, x_goal, y_goal):
        """指定位置まで移動"""
        dx, dy = x_goal - self.x, y_goal - self.y
        # まず IMU ベースで向きを合わせる
        target_theta = math.atan2(dy, dx)
        delta = (target_theta - self.get_yaw() + math.pi) % (2*math.pi) - math.pi
        self.turn(delta)
        # 直進
        dist = math.hypot(dx, dy)
        self.forward(dist)

    def check_layout_update_time(self):
        """レイアウト更新のタイミングをチェック"""
        current_time = time.time()
        if current_time - self.last_layout_update >= LAYOUT_UPDATE_INTERVAL:
            self.last_layout_update = current_time
            return True
        return False

    def load_paths(self):
        """経路データを読み込み"""
        if not self.paths:  # 初回のみ読み込み
            try:
                with open('points.csv') as f:
                    for line in f:
                        s = line.strip()
                        if not s: continue
                        pts = [tuple(map(float,p.split(','))) for p in s.split(';')]
                        self.paths.append([(x/1000.0, y/1000.0) for x,y in pts])
            except FileNotFoundError:
                print("points.csv not found, using default path")
                self.paths = [[(0.5, 0.5), (1.0, 1.0), (0.0, 0.0)]]

    def continue_path_following(self):
        """経路追従を継続（現在の進行状況から）"""
        if not self.paths:
            self.load_paths()
        
        # 現在の経路とポイントから継続
        while self.current_path_index < len(self.paths):
            path = self.paths[self.current_path_index]
            
            # 経路の最初のポイントに移動（ペンを上げて）
            if self.current_point_index == 0:
                print(f"Following path {self.current_path_index+1}/{len(self.paths)}")
                self.pen.write(False)
                self.goto(*path[0])
                self.pen.write(True)
                self.current_point_index = 1
            
            # 残りのポイントを描画
            while self.current_point_index < len(path):
                x, y = path[self.current_point_index]
                self.goto(x, y)
                self.current_point_index += 1
                
                # 定期的にレイアウト推定で位置リセット
                if self.check_layout_update_time():
                    print("Performing periodic layout-based position reset...")
                    return False  # 継続が必要
            
            # この経路が完了
            self.current_path_index += 1
            self.current_point_index = 0
        
        return True  # すべて完了

    def run(self):
        """メインループ"""
        print("Starting robot navigation with layout estimation...")
        
        # 初期状態：レイアウト推定で自己位置を推定
        if self.state == "INITIAL_ESTIMATION":
            print("Performing initial position estimation...")
            if self.reset_position_with_layout():
                self.state = "MOVING_TO_ORIGIN"
            else:
                # 推定失敗時は初期位置のまま
                self.state = "MOVING_TO_ORIGIN"
        
        # 原点に向かって移動
        if self.state == "MOVING_TO_ORIGIN":
            print(f"Moving to origin from ({self.x:.3f}, {self.y:.3f})")
            # ペンを上げてから移動
            self.pen.write(False)
            self.goto(0.0, 0.0)
            self.state = "FOLLOWING_PATH"
            print("Reached origin, starting path following...")
        
        # 経路追従
        if self.state == "FOLLOWING_PATH":
            # 経路追従を継続
            if self.continue_path_following():
                self.state = "COMPLETED"
            else:
                self.state = "LAYOUT_UPDATE"
        
        # レイアウト更新
        if self.state == "LAYOUT_UPDATE":
            if self.reset_position_with_layout():
                self.state = "FOLLOWING_PATH"
            else:
                # 更新失敗時は経路追従を継続
                self.state = "FOLLOWING_PATH"
        
        # 完了
        if self.state == "COMPLETED":
            print("Navigation completed!")
            self.pen.write(False)
            self.stop()
            return
        
        # 次のステップ
        self.step_and_update()
        
        # 状態を継続
        self.run()

if __name__ == '__main__':
    controller = IMUCameraCompassController()
    controller.run()
