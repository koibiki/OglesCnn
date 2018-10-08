package com.example.eglnn.utils;

import com.example.eglnn.Render;
import com.example.eglnn.layer.Layer;

import java.nio.FloatBuffer;

public class DataUtils {

    public static float[] readOutput(Layer layer, FloatBuffer out, int width, int height) {
        Render.transferFromTexture(out, layer.getAttachID(), 0, 0, width, height);
        return out.array();
    }

    /**
     * 标准化输出
     */
    public static float[][][] transform(float[][][] out, float[] array, int width, int height, int channel, int index) {
        for (int c = 0; c < 4; c++) {
            if (4 * index + c < channel) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        out[4 * index + c][h][w] = array[4 * (width * h + w) + c];
                    }
                }
            }
        }
        return out;
    }

}
