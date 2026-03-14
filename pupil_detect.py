import cv2
import numpy as np
import math

class PupilDetector:
    def __init__(self, min_area=20, max_area=800, min_circularity=0.6):
        """
        参考 Pupil Labs 的 2D 暗瞳检测器
        :param min_area: 瞳孔最小面积阈值 (基于像素)
        :param max_area: 瞳孔最大面积阈值 (基于像素)
        :param min_circularity: 最小圆度 (1.0 为完美圆，0.6 可以容忍一定的侧视椭圆变形)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def detect(self, roi_gray):
        """
        在眼部小图中检测瞳孔
        :param roi_gray: 已经裁剪好的眼部灰度图 (如 64x64)
        :return: (best_ellipse, threshold_image) 
                 best_ellipse 格式: ((center_x, center_y), (width, height), angle)
        """
        # 1. 预处理：高斯模糊去噪
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        
        # 2. 动态自适应阈值
        # 寻找图像中最暗的点
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        # 设定阈值：只保留比最暗点稍微亮一点点的区域 (Pupil Labs 会使用更复杂的直方图分析，这里用偏移量平替)
        # 假设瞳孔内的灰度值在极小值之上 25 个灰阶以内
        threshold_value = min_val + 25 
        _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 3. 形态学操作：去除睫毛等细小噪点，填平瞳孔内的反光盲区
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 去除外部噪点
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # 填补内部空洞
        
        # 4. 提取轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        best_ellipse = None
        max_score = 0
        
        # 5. 过滤并寻找最佳瞳孔拟合
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 过滤面积不符的轮廓
            if area < self.min_area or area > self.max_area:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            # 计算圆度
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            # cv2.fitEllipse 至少需要 5 个点
            if circularity > self.min_circularity and len(cnt) >= 5:
                # 拟合椭圆
                ellipse = cv2.fitEllipse(cnt)
                
                # 打分机制：面积适中且越接近完美圆的轮廓得分越高
                score = area * circularity
                if score > max_score:
                    max_score = score
                    best_ellipse = ellipse
                    
        return best_ellipse, thresh

# ================= 测试演示 =================
if __name__ == "__main__":
    # 假设你已经获取到了上一步的 64x64 ROI 小图 (这里创建一个模拟的测试图)
    # 实际使用中，将 eye_crop 传入即可
    simulated_roi = np.ones((64, 64), dtype=np.uint8) * 150
    cv2.circle(simulated_roi, (32, 35), 8, (10, 10, 10), -1) # 模拟暗瞳
    cv2.circle(simulated_roi, (34, 33), 2, (200, 200, 200), -1) # 模拟红外反光点(光斑)
    
    detector = PupilDetector()
    
    # 执行检测
    ellipse, debug_thresh = detector.detect(simulated_roi)
    
    # 可视化结果
    output_img = cv2.cvtColor(simulated_roi, cv2.COLOR_GRAY2BGR)
    if ellipse is not None:
        # 画出拟合的瞳孔椭圆 (绿色)
        cv2.ellipse(output_img, ellipse, (0, 255, 0), 1)
        # 画出瞳孔中心 (红色)
        center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
        cv2.circle(output_img, (center_x, center_y), 1, (0, 0, 255), -1)
        print(f"检测到瞳孔中心: ({center_x}, {center_y})")
    
    # 放大显示以便观察
    cv2.imshow("Original ROI", cv2.resize(simulated_roi, (256, 256)))
    cv2.imshow("Threshold (Debug)", cv2.resize(debug_thresh, (256, 256)))
    cv2.imshow("Pupil Fit", cv2.resize(output_img, (256, 256)))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()