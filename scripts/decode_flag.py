#!/usr/bin/env python3
import sys

# OpenCV bit mask constants
CV_CN_SHIFT = 3
CV_DEPTH_MAX = 1 << CV_CN_SHIFT
CV_CN_MAX = 512
CV_MAT_DEPTH_MASK = CV_DEPTH_MAX - 1
CV_MAT_CN_MASK = (CV_CN_MAX - 1) << CV_CN_SHIFT
CV_MAT_CONT_FLAG_SHIFT = 14
CV_MAT_CONT_FLAG = 1 << CV_MAT_CONT_FLAG_SHIFT
CV_SUBMAT_FLAG = 1 << 15

DEPTH_STR = {
    0: "CV_8U",
    1: "CV_8S",
    2: "CV_16U",
    3: "CV_16S",
    4: "CV_32S",
    5: "CV_32F",
    6: "CV_64F",
}

def decode_flags(f: int):
    depth = f & CV_MAT_DEPTH_MASK
    channels = ((f & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1
    cont = bool(f & CV_MAT_CONT_FLAG)
    submat = bool(f & CV_SUBMAT_FLAG)
    type_str = f"{DEPTH_STR.get(depth, '?')}C{channels}"
    return depth, channels, cont, submat, type_str

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python decode_flag.py <flags_degeri>")
        sys.exit(1)
    
    f = int(sys.argv[1])
    depth, ch, cont, sub, tstr = decode_flags(f)

    print(f"flags = {f} (0b{f:032b})")
    print(f"Depth      : {DEPTH_STR.get(depth, '?')} ({depth})")
    print(f"Channels   : {ch}")
    print(f"Continuous : {cont}")
    print(f"Submatrix  : {sub}")
    print(f"Type       : {tstr}")
