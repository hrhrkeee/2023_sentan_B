import math

print("rが正規化された、ボールの極座標を取得します。")

# ボードの (x,y) 座標と半径Rを保存するインスタンスを生成するクラス
class Board_Info :
    
    def __init__(self , arg_x : int , arg_y : int , arg_radius : int) :
        self.x = arg_x
        self.y = arg_y
        self.radius = arg_radius
        
# ボール中心の (x,y) 座標を保存するインスタンスを生成するクラス
class Ball_Info :
    
    def __init__(self , arg_x : int , arg_y : int) :
        self.x = arg_x
        self.y = arg_y

# ボールの (r,θ) 極座標を保存するインスタンスを生成するクラス
class Polar_Coordinate :
    
    def __init__(self , arg_r : float , arg_θ : float) :
        self.r = arg_r
        self.θ = arg_θ

number_of_object = int(input("\n検知した物体の総数 > "))

ball_center_list = []

for _ in range(number_of_object) :
    print("\nボール -> 1 / ボード -> 2")
    object_label = int(input("> "))  # 物体検知で入力

    # ラベルが「2」の時 (ボード)
    if  (object_label == 2) : 
        board_left_up_x = int(input("\nボード|左上|x座標 > "))  # 物体検知で入力
        board_left_up_y = int(input("ボード|左上|y座標 > "))  # 物体検知で入力
    
        board_right_down_x = int(input("ボード|右下|x座標 > "))  # 物体検知で入力
        board_right_down_y = int(input("ボード|右下|y座標 > "))  # 物体検知で入力
    
        board_center_x = (board_left_up_x + board_right_down_x) / 2
        board_center_y = (board_left_up_y + board_right_down_y) / 2
        
        board_radius = board_right_down_x - board_center_x
        
        # ボードの中心(x,y)座標と半径を保存
        board_info = Board_Info(board_center_x , board_center_y , board_radius)

    # ラベルが「1」の時 (ボール)
    elif (object_label == 1) : 
        
        ball_count = 1 
        print(f"\n{ball_count}個目のールの座標")
        ball_left_up_x = int(input("ボール|左上|x座標 > "))  # 物体検知で入力
        ball_left_up_y = int(input("ボール|左上|y座標 > "))  # 物体検知で入力 
        
        ball_right_down_x = int(input("ボール|右下|x座標 > "))  # 物体検知で入力
        ball_right_down_y = int(input("ボール|右下|y座標 > "))  # 物体検知で入力
        
        ball_center_x = (ball_left_up_x + ball_right_down_x) / 2 
        ball_center_y = (ball_left_up_y + ball_right_down_y) / 2
      
        # ボールの中心(x,y)座標を保存
        ball_center = Ball_Info(ball_center_x , ball_center_y)
        ball_center_list.append(ball_center)
        
        ball_count += 1
        
else :
    print("\nボードと全ボールの(x,y)を保存しました。")

ball_polar_coordinate_list = []
for i in range(len(ball_center_list)) :
    
    # ボード中心とボール中心のX座標の差分を計算
    difference_x_coordinate =abs(board_info.x - ball_center_list[i].x)
    # ボード中心とボール中心のY座標の差分を計算
    difference_y_coordinate = abs(board_info.y - ball_center_list[i].y)
    # 上記2つの長さから、三平方の定理を用いてrを計算
    r = math.sqrt(difference_x_coordinate**2 + difference_y_coordinate**2)
    # 上記で求めたrを、ボードの半径を用いて正規化
    normalized_r = r / board_info.radius
    
    # 三角関数cosの値を求める (x -> 底辺 / r -> 斜辺)
    cos_value = difference_x_coordinate / r 
    # 上記で求めた値から、θを逆算(ラジアン出力)
    θ = math.acos(cos_value)
    #　上記で求めたθを、°に変換
    θ_degree = math.degrees(θ)
    
    # ボールが第2象限にある時
    if   ((board_info.x > ball_center_list[i].x) and (board_info.y > ball_center_list[i].y)) :
        θ_degree = 180 - θ_degree
    # ボールが第3象限にある時
    elif ((board_info.x > ball_center_list[i].x) and (board_info.y < ball_center_list[i].y)) :
        θ_degree += 180 
    # ボールが第4象限にある時
    elif ((board_info.x < ball_center_list[i].x) and (board_info.y < ball_center_list[i].y)) : 
        θ_degree = 360 - θ_degree
    
    polar_coordinate = Polar_Coordinate(normalized_r , θ_degree) 
    ball_polar_coordinate_list.append(polar_coordinate)
else :
    print("全ボールの、正規化済み極座標を生成しました。")

print() 
for i in range(len(ball_center_list)) : 
    print(f"{i+1}個目のボールの極座標")
    print(f"r -> {ball_polar_coordinate_list[i].r}")
    print(f"θ -> {ball_polar_coordinate_list[i].θ}")
    
    
    