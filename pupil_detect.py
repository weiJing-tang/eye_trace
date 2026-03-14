import cv2
import numpy as np
import math

class PupilDetector:
    """微观提取器：负责在 64x64 的 ROI 内精确定位瞳孔"""
    def __init__(self, min_area=10, max_area=800, min_circularity=0.64):
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def detect(self, roi_gray):
        enhanced = self.clahe.apply(roi_gray)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        offset = max(15, (max_val - min_val) * 0.15) 
        threshold_value = min_val + offset
        
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
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
                
                # 中心亮度校验
                cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
                if 0 <= cx < blurred.shape[1] and 0 <= cy < blurred.shape[0]:
                    if blurred[cy, cx] > threshold_value + 20:
                        continue 
                
                score = area * circularity
                if score > max_score:
                    max_score = score
                    best_ellipse = ellipse
                    
        return best_ellipse, thresh

class SingleEyeTracker:
    """单目标状态机"""
    def __init__(self, start_center, roi_size=64, max_fails=5):
        self.center = start_center
        self.roi_size = roi_size
        self.half_roi = roi_size // 2
        self.max_fails = max_fails
        self.fail_count = 0
        self.is_active = True
        self.global_ellipse = None
        self.local_ellipse = None # 用于在 64x64 小图上绘制
        self.eye_crop = None      # 存储 64x64 截取原图

    def get_bbox(self, frame_shape):
        h, w = frame_shape[:2]
        cx, cy = self.center
        x1 = max(0, cx - self.half_roi)
        y1 = max(0, cy - self.half_roi)
        x2 = min(w, cx + self.half_roi)
        y2 = min(h, cy + self.half_roi)
        return (x1, y1, x2, y2)

class MultiEyeTrackerPipeline:
    """多目标追踪管线 (增加人脸约束与小图输出)"""
    def __init__(self, roi_size=64, max_fails=5, merge_distance=40):
        self.roi_size = roi_size
        self.max_fails = max_fails
        self.merge_distance = merge_distance
        self.trackers = []
        

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 推荐使用带眼镜检测的眼部级联模型，鲁棒性稍好
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        self.pupil_detector = PupilDetector()

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        extracted_crops = [] 
        
        # ==========================================
        # 1. 宏观检测：人脸约束与黄金“眼带”裁剪
        # ==========================================
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))
        
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (50, 50, 50), 1)
            
            # 砍掉人脸顶部 22% (额头和眉毛)，以及底部 45% (鼻子和嘴巴)
            start_y = fy + int(fh * 0.22)
            end_y = fy + int(fh * 0.55)
            roi_face_gray = gray[start_y:end_y, fx:fx + fw]
            
            # 画出我们在人脸上实际搜索眼睛的区域（绿色虚线框）
            cv2.rectangle(display_frame, (fx, start_y), (fx + fw, end_y), (0, 255, 0), 1, cv2.LINE_AA)
            
            eyes = self.eye_cascade.detectMultiScale(roi_face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (ex, ey, ew, eh) in eyes:
                # 坐标转换
                global_ex = fx + ex
                global_ey = start_y + ey 
                new_cx = global_ex + ew // 2
                new_cy = global_ey + eh // 2
                
                is_new = True
                for t in self.trackers:
                    dist = math.hypot(t.center[0] - new_cx, t.center[1] - new_cy)
                    if dist < self.merge_distance:
                        is_new = False
                        break
                        
                if is_new:
                    self.trackers.append(SingleEyeTracker((new_cx, new_cy), self.roi_size, self.max_fails))
        # ==========================================
        # 2. 微观提取与小图截取
        # ==========================================
        for t in self.trackers:
            bbox = t.get_bbox(gray.shape)
            if bbox is None:
                t.is_active = False
                continue
                
            x1, y1, x2, y2 = bbox
            roi_gray = gray[y1:y2, x1:x2]
            
            if roi_gray.shape[0] < 20 or roi_gray.shape[1] < 20:
                t.is_active = False
                continue
                
            # 截取用于展示的彩色小图，并统一缩放为 64x64 
            crop_color = frame[y1:y2, x1:x2].copy()
            crop_color = cv2.resize(crop_color, (self.roi_size, self.roi_size))
            
            ellipse, _ = self.pupil_detector.detect(roi_gray)
            
            if ellipse is not None:
                t.fail_count = 0
                local_cx, local_cy = ellipse[0]
                global_cx = int(x1 + local_cx)
                global_cy = int(y1 + local_cy)
                
                t.global_ellipse = ((global_cx, global_cy), ellipse[1], ellipse[2])
                t.local_ellipse = ellipse # 保存局部椭圆，用于在 64x64 图上绘制
                t.center = (global_cx, global_cy)
                
                # 在 64x64 截取图上绘制瞳孔特征
                cv2.ellipse(crop_color, t.local_ellipse, (0, 255, 0), 1)
                cv2.circle(crop_color, (int(local_cx), int(local_cy)), 2, (0, 0, 255), -1)
            else:
                t.fail_count += 1
                t.global_ellipse = None
                t.local_ellipse = None
                if t.fail_count >= t.max_fails:
                    t.is_active = False
            
            # 如果追踪器没死，就把小图加入展示队列
            if t.is_active:
                extracted_crops.append(crop_color)
                    
        # ==========================================
        # 3. 清理与去重
        # ==========================================
        self.trackers = [t for t in self.trackers if t.is_active]
        active_trackers = []
        for i, t1 in enumerate(self.trackers):
            overlap = False
            for t2 in active_trackers:
                if math.hypot(t1.center[0] - t2.center[0], t1.center[1] - t2.center[1]) < self.merge_distance:
                    overlap = True
                    break
            if not overlap:
                active_trackers.append(t1)
        self.trackers = active_trackers

        # ==========================================
        # 4. 绘制大图输出
        # ==========================================
        for i, t in enumerate(self.trackers):
            bbox = t.get_bbox(gray.shape)
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 150, 0), 1)
                cv2.putText(display_frame, f"ID:{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
                
            if t.global_ellipse:
                cv2.ellipse(display_frame, t.global_ellipse, (0, 255, 0), 1)
                cv2.circle(display_frame, t.global_ellipse[0], 2, (0, 0, 255), -1)

        cv2.putText(display_frame, f"Tracking Eyes: {len(self.trackers)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display_frame, extracted_crops


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pipeline = MultiEyeTrackerPipeline(roi_size=64, max_fails=5)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 返回处理后的大图，以及包含所有 64x64 小图的列表
        display_frame, extracted_crops = pipeline.process_frame(frame)
        
        cv2.imshow("Multi-Eye Tracking Pipeline", display_frame)

        # 动态拼接并展示 64x64 的 ROI 小图
        if extracted_crops:
            # 将所有小图水平拼接成一张长条图
            stacked_crops = np.hstack(extracted_crops)
            # 为了方便观看，稍微放大一点 (可选)
            stacked_crops_display = cv2.resize(stacked_crops, (stacked_crops.shape[1] * 2, stacked_crops.shape[0] * 2))
            cv2.imshow("Extracted Eyes (64x64 ROI)", stacked_crops_display)
        else:
            # 如果当前没有追踪到眼睛，可以销毁小图窗口或显示一张纯黑图防报错
            try:
                cv2.destroyWindow("Extracted Eyes (64x64 ROI)")
            except:
                pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()