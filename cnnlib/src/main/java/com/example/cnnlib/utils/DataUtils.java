package com.example.cnnlib.utils;

import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.render.ComputeRender;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

public class DataUtils {

    private static final String TAG = "DataUtils";

    /**
     * 测试用输入数据
     * */
    public static float[][][] createInputBuffer(int[] shape) {
        int channel = shape[2];
        int width = shape[0];
        int height = shape[1];
        float input[][][] = new float[channel][height][width];
        for (int c = 0; c < channel; c++) {
            for (int w = 0; w < width; w++) {
                for (int h = 0; h < height; h++) {
                    input[c][h][w] = channel * (h * width + w) + c;
                }
            }
        }
        return input;
    }

    public static float[][][] readOutput(Layer layer) {
        int[] outputShape = layer.getOutputShape();
        int width = outputShape[0];
        int height = outputShape[1];
        int channel = outputShape[2];
        float[][][] out = new float[channel][height][width];

        int count = NetUtils.alignBy4(channel);
        for (int i = 0; i < count / 4; i++) {
            int[] indexes = NetUtils.getXYIndex(width, i, Constants.S_TEXTURE_SIZE);
            int startX = indexes[0] * width;
            int startY = indexes[1] * height;
            FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
            allocate = (FloatBuffer) ComputeRender.transferFromTexture(allocate, layer.getAttachID(), startX, startY, width, height);
            float[] array = allocate.array();
            out = transform(out, array, width, height, i, channel);
        }
        return out;
    }

    /**
     * 标准化输出
     * */
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

    /**
     * 测试用全连接 kennel
     * */
    public static float[] createFullConnKennel(int alignSize, int kennelSize, int index) {
        float[] kennel = new float[alignSize];       // 最后一位是bias
        for (int i = 0; i < kennelSize - 1; i++) {
            kennel[i] = i ;
        }
        kennel[alignSize - 1] = 1;
        return kennel;
    }

    /**
     * 测试用conv kennel
     * channel需要4对齐, 依次排满通道后再排下一个值, 最后4个值为 bias, 0, 0, 0
     */
    public static float[] createConvKennel(int num, int width, int height, int channel, float bias) {
        int alignChannel = NetUtils.alignBy4(channel);
        float[] data = new float[width * height * alignChannel + 4];
        for (int i = 0; i < width * height; i++) {
            for (int ii = 0; ii < channel; ii++) {
                data[i * alignChannel + ii] = 1;
            }
        }
        data[width * height * alignChannel] = bias;
        return data;
    }

}
