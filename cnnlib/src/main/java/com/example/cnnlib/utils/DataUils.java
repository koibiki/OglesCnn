package com.example.cnnlib.utils;

import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.render.ComputeRender;

import java.nio.FloatBuffer;
import java.util.Random;

public class DataUils {

    public static float[][] createInputBuffer(int[] shape) {
        int channel = shape[2];

        int r = new Random().nextInt(10);
        int areaCapacity = shape[0] * shape[1];
        float input[][] = new float[channel][areaCapacity];
        for (int j = 0; j < channel; j++) {
            for (int i = 0; i < areaCapacity; i++) {
                input[j][i] = i * r * 0.1f;
            }
        }
        return input;
    }

    public static void readOutput(Layer layer) {
        int channel = layer.getOutputShape()[2];
        int[] count = SortUtils.getCount(channel);
        for (int i = 0; i < count[0]; i++) {
            readOutput(layer, i);
        }
    }

    private static void readOutput(Layer layer, int index) {
        int[] outputShape = layer.getOutputShape();
        int width = outputShape[0];
        int[] indexes = SortUtils.getXYIndex(width, index);
        int height = outputShape[1];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
        allocate = (FloatBuffer) ComputeRender.transferFromTexture(allocate, layer.getAttachID(), startX, startY, width, height);
        float[] array = allocate.array();
//        LogUtils.printf(array, width, "output" + indexes[0] + "_" + indexes[1] + ".txt");
    }

    public static float[] createFullConnKennel(int size, int index) {
        float[] kennel = new float[size];       // 最后一位是bias
        for (int i = 0; i < size - 1; i++) {
            kennel[i] = 1f;
        }
        kennel[size - 1] = 0;
        return kennel;
    }

    public static float[] createConvKennel(int num, int length, int channel, float bias) {
        float[] data = new float[length * length * channel + 1];
        for (int i = 0; i < length * length * channel; i++) {
            data[i] = 0.1f;
        }
        data[length * length * channel] = bias;
        return data;
    }

}
