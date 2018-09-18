package com.example.eglnn.utils;

import android.opengl.GLES31;

public class Utils {

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

    public static int getFormat(int c) {
        if (c == 1) {
            return GLES31.GL_R32F;
        } else if (c == 2) {
            return GLES31.GL_RG32F;
        } else if (c == 3) {
            return GLES31.GL_RGB32F;
        } else {
            return GLES31.GL_RGBA32F;
        }
    }


}
