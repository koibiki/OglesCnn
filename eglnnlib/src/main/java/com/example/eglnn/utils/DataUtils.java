package com.example.eglnn.utils;

import com.example.eglnn.Render;
import com.example.eglnn.layer.Layer;

import java.nio.FloatBuffer;

public class DataUtils {

    private static final String TAG = "DataUtils";

    public static float[][][] readOutput(Layer layer) {
        int[] outputShape = layer.getOutputShape();
        int width = outputShape[0];
        int height = outputShape[1];
        int channel = outputShape[2];
        float[][][] out = new float[channel][height][width];

        int count = Utils.alignBy4(channel);
        for (int i = 0; i < 1; i++) {
            FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
            allocate = (FloatBuffer) Render.transferFromTexture(allocate, layer.getAttachID(), 0, 0, width, height);
            float[] array = allocate.array();
            out = transform(out, array, width, height, i, channel);
        }
        return out;
    }

    /**
     * 标准化输出
     */
    private static float[][][] transform(float[][][] out, float[] array, int width, int height, int index, int channel) {
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
