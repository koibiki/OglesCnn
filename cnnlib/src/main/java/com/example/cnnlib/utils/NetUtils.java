package com.example.cnnlib.utils;

public class NetUtils {

    // 获得当前特征图在纹理上的坐标
    public static int[] getXYIndex(int width, int i, int texWidth) {
        int xIndex = i < texWidth / width ? i : i % (texWidth / width);
        int yIndex = i / (texWidth / width);
        return new int[]{xIndex, yIndex};
    }

    // 按照 4 对齐
    public static int alignBy4(int channel) {
        final int align = 4;
        return (channel + (align - 1)) & ~(align - 1);
    }


}
