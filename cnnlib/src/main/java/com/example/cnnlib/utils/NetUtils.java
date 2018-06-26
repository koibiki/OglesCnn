package com.example.cnnlib.utils;

public class NetUtils {

    // 获得该通道总数下数据需要的特征图 存储个数
    public static int[] getCount(int channel) {
        int count = channel / 4;
        int remain = channel % 4;
        if (remain != 0) {
            count = count + 1;
        }
        return new int[]{count, remain};
    }

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
