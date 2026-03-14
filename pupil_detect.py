import cv2
import numpy as np
import math

class PupilDetector:
    """微观提取器：负责在 64x64 的 ROI 内精确定位瞳孔"""
    def __init__(self, min_area=10, max_area=800, min_circularity=0.5):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def detect(self, roi_gray):
        # 高斯去噪与动态阈值
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        threshold_value = min_val + 25 
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 提取并过滤轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best_ellipse = None
        max_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area: continue
            
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
                
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            if circularity > self.min_circularity and len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                score = area * circularity
                if score > max_score:
                    max_score = score
                    best_ellipse = ellipse
                    
        return best_ellipse, thresh

class EyeTrackerPipeline:
    """系统状态机：宏观框定 + 微观提取 + 自适应回退"""
    def __init__(self, roi_size=64, max_fails=5):
        self.roi_size = roi_size
        self.half_roi = roi_size // 2
        self.max_fails = max_fails  # 连续 N 帧失败则回退
        
        self.state = 'DETECTING'    # 初始状态
        self.fail_count = 0
        self.roi_center = None      # 记录当前 ROI 的中心点全局坐标
        
        # 加载宏观检测器
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # 加载微观提取器
        self.pupil_detector = PupilDetector()

    def _get_roi_bbox(self, frame_shape):
        """根据当前的中心点，计算 64x64 的安全截取边界"""
        if self.roi_center is None: return None
        h, w = frame_shape[:2]
        cx, cy = self.roi_center
        
        x1 = max(0, cx - self.half_roi)
        y1 = max(0, cy - self.half_roi)
        x2 = min(w, cx + self.half_roi)
        y2 = min(h, cy + self.half_roi)
        
        return (x1, y1, x2, y2)

    def process_frame(self, frame):
        """处理主流程"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        global_ellipse = None
        roi_debug = None
        
        # -----------------------------------------
        # 1. 宏观框定 (Macro Detection)
        # -----------------------------------------
        if self.state == 'DETECTING':
            # 在全图中检测眼睛
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            if len(eyes) > 0:
                # 获取第一个眼睛的边界框 (x, y, w, h)
                ex, ey, ew, eh = eyes[0]
                # 将眼睛的几何中心设为初始 ROI 中心
                self.roi_center = (ex + ew // 2, ey + eh // 2)
                self.state = 'TRACKING'
                self.fail_count = 0
            else:
                return frame, None, self.state

        # -----------------------------------------
        # 2. 微观提取与跟踪 (Micro Extraction & Implicit Tracking)
        # -----------------------------------------
        if self.state == 'TRACKING':
            bbox = self._get_roi_bbox(gray.shape)
            if bbox is None:
                self.state = 'DETECTING'
                return frame, None, self.state
                
            x1, y1, x2, y2 = bbox
            roi_gray = gray[y1:y2, x1:x2]
            
            if roi_gray.shape[0] < 10 or roi_gray.shape[1] < 10:
                self.state = 'DETECTING'
                return frame, None, self.state
                
            roi_debug = roi_gray.copy()
            
            # 在 64x64 区域内运行瞳孔检测
            ellipse, _ = self.pupil_detector.detect(roi_gray)
            
            if ellipse is not None:
                # 提取成功，重置失败计数器
                self.fail_count = 0
                
                # ellipse[0] 是局部坐标系(64x64)下的中心点 (local_cx, local_cy)
                local_cx, local_cy = ellipse[0]
                
                # 转换为全图的全局坐标系
                global_cx = int(x1 + local_cx)
                global_cy = int(y1 + local_cy)
                
                # 重构用于在全图绘制的椭圆参数
                global_ellipse = ((global_cx, global_cy), ellipse[1], ellipse[2])
                
                # 【核心优化】：将找到的瞳孔中心，更新为下一帧 64x64 截取框的新中心！
                # 这样只要瞳孔没有在一帧内飞出 64x64 的范围，框就会永远跟着瞳孔走。
                self.roi_center = (global_cx, global_cy)
                
            # -----------------------------------------
            # 3. 自适应回退 (Adaptive Fallback)
            # -----------------------------------------
            else:
                self.fail_count += 1
                if self.fail_count >= self.max_fails:
                    # 连续 N 帧丢失（比如眨眼、转头），退回全图宏观检测
                    self.state = 'DETECTING'
                    self.roi_center = None

        # 可视化绘制
        display_frame = frame.copy()
        
        # 画出 64x64 的跟踪 ROI 框
        if self.roi_center is not None and self.state == 'TRACKING':
            bbox = self._get_roi_bbox(gray.shape)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # 画出瞳孔中心与拟合的椭圆
        if global_ellipse is not None:
            cv2.ellipse(display_frame, global_ellipse, (0, 255, 0), 1)
            cv2.circle(display_frame, global_ellipse[0], 2, (0, 0, 255), -1)
            
        cv2.putText(display_frame, f"State: {self.state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display_frame, roi_debug, self.state

# ================= 运行示例 =================
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    # 初始化管线，设定截取框大小为 64，容忍 5 帧的连续丢失
    pipeline = EyeTrackerPipeline(roi_size=64, max_fails=5)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 传入全图，传出绘制好结果的全图和 64x64 的 ROI debug 图
        display_frame, roi_debug, current_state = pipeline.process_frame(frame)

        cv2.imshow("IR Eye Tracking Pipeline", display_frame)
        if roi_debug is not None:
            # 放大显示 64x64 的局部 ROI，方便观察
            cv2.imshow("64x64 ROI", cv2.resize(roi_debug, (128, 128)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()