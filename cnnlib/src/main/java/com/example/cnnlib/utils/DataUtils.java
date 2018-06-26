package com.example.cnnlib.utils;

import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.render.ComputeRender;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DataUtils {

    private static final String TAG = "DataUtils";

    public static float[][] createInputBuffer(int[] shape) {
        int channel = shape[2];

        int r = new Random().nextInt(10);
        int areaCapacity = shape[0] * shape[1];
        float input[][] = new float[channel][areaCapacity];
        for (int j = 0; j < channel; j++) {
            for (int i = 0; i < areaCapacity; i++) {
                input[j][i] = i * r * 0.0001f;
            }
        }
        return input;
    }

    public static List<float[]> readOutput(Layer layer) {
        List<float[]> results = new ArrayList<>();
        int channel = layer.getOutputShape()[2];
        int[] count = NetUtils.getCount(channel);
        for (int i = 0; i < count[0]; i++) {
            results.add(readOutput(layer, i));
        }
        return results;
    }

    private static float[] readOutput(Layer layer, int index) {
        int[] outputShape = layer.getOutputShape();
        int width = outputShape[0];
        int[] indexes = NetUtils.getXYIndex(width, index, Constants.S_TEXTURE_SIZE);
        int height = outputShape[1];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
        allocate = (FloatBuffer) ComputeRender.transferFromTexture(allocate, layer.getAttachID(), startX, startY, width, height);
        float[] array = allocate.array();
//        LogUtils.printf(array, width, "output" + indexes[0] + "_" + indexes[1] + ".txt");
        return array;
    }

    public static float[] createFullConnKennel(int size, int index) {
        float[] kennel = new float[size];       // 最后一位是bias
        for (int i = 0; i < size - 1; i++) {
            kennel[i] = 1f;
        }
        kennel[size - 1] = 0;
        return kennel;
    }

    public static float[] createConvKennel(int num, int width, int height, int channel, float bias) {
        float[] data = new float[width * height * channel + 1];
        for (int i = 0; i < width * height; i++) {
            for (int ii = 0; ii < channel; ii++) {
                data[i] = i + 0.00001f * ii;
            }
        }
        data[width * height * channel] = bias;
        return data;
    }

    /**
     * channel需要4对齐, 依次排满通道后再排下一个值, 最后4个值为 bias, 0, 0, 0
     * */
    public static float[] createConvKennel2(int num, int width, int height, int channel, float bias) {
        int alignChannel = NetUtils.alignBy4(channel);
        float[] data = new float[width * height * alignChannel + 4];
        for (int i = 0; i < width * height; i++) {
            for (int ii = 0; ii < channel; ii++) {
                data[i * alignChannel + ii] = i + 0.00001f * ii;
            }
        }
        data[width * height * channel] = bias;
        return data;
    }

}
