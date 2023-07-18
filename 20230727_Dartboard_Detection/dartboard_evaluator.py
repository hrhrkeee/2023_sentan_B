import math, random
import numpy as np
from typing import Any
from ultralytics import YOLO

from pathlib import Path


class DartboardEvaluator:
    def __init__(self, yolo_model_path):

        self.yolo_model_path = yolo_model_path
        self.yolo_model      = YOLO(str(self.yolo_model_path))

        # YOLO検出の設定
        self.detect_confidence_threshold = 0.5
        self.detect_IoU = 0.001
        self.detect_max_instance = 300
        self.detect_source_augment = False
        
        # ボードの情報
        self.board_radius_divisions:int = 3     # ボードの円の数
        self.board_circle_divisions:int = 12    # ボードの円の分割数
        self.board_each_circle_radius:list[int] = [0, 0.2, 0.6, 1.0]   # ボードの各円の半径割合

        # ボードの点数表
        score_table1 = [200]
        score_table2 = [160, 100, 150, 90, 140, 80, 190, 130, 180, 120, 170, 110]
        score_table3 = [40, 10, 30, 10, 20, 10, 70, 10, 60, 10, 50, 10]
        score_table  = [score_table1, score_table2, score_table3]
        # score_tableの中の配列の長さをcircle_divisionsまで拡張する
        self.board_score_table:list = [(table*(math.ceil(self.board_circle_divisions/len(table))))[:12] for table in score_table]

    def __call__(self, input_img:np.ndarray) -> Any:
        print("__call__")
        pass

    def detect(self, input_img:np.ndarray) -> Any:
        result = self.yolo_model(
                source    = input_img,
                conf      = self.detect_confidence_threshold,
                iou       = self.detect_IoU,
                max_det   = self.detect_max_instance,
                augment   = self.detect_source_augment,
                save      = False,
                classes   = None, # [1, 2, 3],
            )

        return result

    def score_calculate(self, ball_distance:float, ball_radian:float) -> int:
        """
        ball_distance: ボードの中心とダーツの距離 (0.0 ~ 1.0)
        ball_radian:   ボードの中心座標からダーツまでの角度[rad]  (0 ~ 360)
        """

        score = 0
        # ダーツがボードの外にある場合
        if ball_distance > 1.0:
            return score
        
        # ball_radianが0~360の範囲に収まるようにする
        if ball_radian < 0 or  ball_radian > 360:
            ball_radian %= 360
        
        base_angle = float(360/self.board_circle_divisions) 
        
        for j in range(self.board_radius_divisions) :
            if (self.board_each_circle_radius[j] <= ball_distance < self.board_each_circle_radius[j+1]) : # ダーツの位置を、rから判断
                print("darts_distanceは{}番目の円の範囲にありました。" .format(j+1))  
            
                for k in range(self.board_circle_divisions) : 
                    if(base_angle * k <= ball_radian < base_angle * (k+1)) :  # ダーツの位置を、θから判断
                        print("θは{}°以内の位置にありました。" .format(base_angle * (k+1)))
                        return self.board_score_table[j][k]          

        return score

    def ball_coordinate_detect(self):
        ball_distance = 0
        ball_radian = 0

        return ball_distance, ball_radian