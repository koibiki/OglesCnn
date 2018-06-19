package com.example.cnnlib.utils;

import com.example.cnnlib.layer.Layer;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;

import java.nio.FloatBuffer;

public class DataUils {

    public static float[][] createInputBuffer(LayerParams layerParams) {
        int[] inputShape = layerParams.inputShape;
        int channel = inputShape[2];
        int areaCapacity = inputShape[0] * inputShape[1];
        float input[][] = new float[channel][areaCapacity];
        for (int j = 0; j < channel; j++) {
            for (int i = 0; i < areaCapacity; i++) {
                input[j][i] = i;
            }
        }
        return input;
    }

    public static void readOutput(Layer layer) {
        int channel = layer.getLayerParams().outputShape[2];
        int[] count = SortUtils.getCount(channel);
        for (int i = 0; i < count[0]; i++) {
            readOutput(layer, i);
        }
    }

    private static void readOutput(Layer layer, int index) {
        LayerParams layerParams = layer.getLayerParams();
        int width = layerParams.outputShape[0];
        int[] indexes = SortUtils.getXYIndex(width, index);
        int height = layerParams.outputShape[1];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
        allocate = (FloatBuffer) ComputeRender.transferFromTexture(allocate, layer.getAttachID(), startX, startY, width, height);
        float[] array = allocate.array();
        // LogUtils.printf(array, width, "output" + indexes[0] + "_" + indexes[1] + ".txt");
    }

    public static float[] createFullConnKennel(int size, int index) {
        float[] kennel = new float[size];       // 最后一位是bias
        for (int i = 0; i < size - 1; i++) {
            kennel[i] = 1;
        }
        kennel[size - 1] = 0.1f * index;
        return kennel;
    }

}
