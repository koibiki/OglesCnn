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
                input[j][i] = i ;
            }
        }
        return input;
    }


    public static void readOutput(Layer layer) {
        int count = SortUtils.getCount(layer.getLayerParams());
        for (int i = 0; i < count; i++) {
            readOutput(layer, i);
        }
    }


    private static void readOutput(Layer layer, int index) {
        LayerParams layerParams = layer.getLayerParams();
        int[] indexes = SortUtils.getXYIndex(layerParams, index);
        int width = layerParams.outputShape[0];
        int height = layerParams.outputShape[1];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        FloatBuffer allocate = FloatBuffer.allocate(width * height * 4);
        allocate = (FloatBuffer) ComputeRender.transferFromTexture(allocate, layer.getAttachID(), startX, startY, width, height);
        float[] array = allocate.array();
        // LogUtils.printf(array, width, "output" + indexes[0] + "_" + indexes[1] + ".txt");
    }

}
