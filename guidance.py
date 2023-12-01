import random

def gd(ele, azi):
    angle_offset1 = 0 # 5 * random.uniform(0, 1)
    angle_offset2 = 0 # 3 * random.uniform(0, 1)
    azimuth = angle_offset1 + azi
    azimuth = round(azimuth, 2)
    elevation = angle_offset2 + ele
    elevation = round(elevation, 2)
    return(elevation, azimuth)

# gud_e, gud_a = gd(35.26, 45.0)
# print(f"{gud_e, gud_a}")