from enum import Flag
import numpy as np
import math

N1 = 12
N2 = 6
N3 = 1.5
A = 0.6 # 无人机的尺寸为0.6*0.6
U = 0.00000001668
def assit(focal_length):
    D1 = N1 * U
    D1 = math.sqrt(D1)
    R1 = (A * focal_length) / D1
    D2 = N2 * U
    D2 = math.sqrt(D2)
    R2 = (A * focal_length) / D2
    D3 = N3 * U
    D3 = math.sqrt(D3)
    R3 = (A * focal_length) / D3
    return R1, R2, R3

class CaptureState(Flag):
    out_of_horizontal_or_vertial = 0,   
    out_of_focal = 1,
    detecting_target = 2,
    recognizing_target = 4,
    identifying_target = 8   

    @classmethod
    def get_capture_state(cls, defender_azimuth_low, defender_azimuth_high, azimuth_deg,
                          defender_elevation_low, defender_elevation_high, elevation_deg,
                          distance, focal_length):
        p1 = 1 if defender_azimuth_low <= azimuth_deg <= defender_azimuth_high else 0
        p2 = 1 if defender_elevation_low <= elevation_deg <= defender_elevation_high else 0 
        if not p1 or not p2: 
            return cls.out_of_horizontal_or_vertial 

        # 计算两点的间距
        R = distance
        d = (A * focal_length) / distance
        d = d**2
        n = d / U
        if n > N1:
            return cls.identifying_target # 辨认目标
        else:
            if n > N2:
                return cls.recognizing_target # 识别目标
            else:
                if n > N3:
                    return cls.detecting_target # 探测目标
                else:
                    return cls.out_of_focal # 

if __name__ == "__main__":
    z1, z2, z3 = assit(0.129)
    print(f"{z1, z2, z3}")