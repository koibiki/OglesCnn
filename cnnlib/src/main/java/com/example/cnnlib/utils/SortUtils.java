package com.example.cnnlib.utils;

import com.example.cnnlib.model.LayerParams;

public class SortUtils {

    public static int getCount(LayerParams layerParams) {
        int channel = layerParams.outputShape[2];
        int count = channel / 4;
        int remain = channel % 4;
        if (remain != 0) {
            count = count + 1;
        }
        return count;
    }

    public static int[] getXYIndex(LayerParams layerParams, int i) {
        int width = layerParams.outputShape[0];
        int xIndex = i < 1024 / width ? i : i % (1024 / width);
        int yIndex = i / (1024 / width);
        return new int[]{xIndex, yIndex};
    }
}
