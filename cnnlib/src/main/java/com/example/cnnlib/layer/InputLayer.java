package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.SortUtils;

import java.nio.FloatBuffer;

public class InputLayer extends Layer {

    public InputLayer(Context context, LayerParams layerParams) {
        super(context, layerParams);
    }

    private float[][] createInputBuffer() {
        int[] inputShape = mLayerParams.inputShape;
        int channel = inputShape[2];
        int areaCapacity = inputShape[0] * inputShape[1];
        float input[][] = new float[channel][areaCapacity];

        for (int j = 0; j < channel; j++) {
            for (int i = 0; i < areaCapacity; i++) {
                input[j][i] = i % inputShape[0];
            }
        }
        return input;
    }

    @Override
    public int forwardProc(int InputTexID, int attachID) {
        float[][] input = createInputBuffer();
        int texture = ComputeRender.createTexture(attachID);
        int count = SortUtils.getCount(mLayerParams);
        for (int i = 0; i < count; i++) {
            writeInput(input, texture, i);
        }
        //ComputeRender.transferToTexture(inputBuffer, texture, 0, 0, outputShape[0], outputShape[1]);
        mOutputTexID = texture;
        return mOutputTexID;
    }

    private void writeInput(float[][] input, int texID, int index) {
        int[] indexes = SortUtils.getXYIndex(mLayerParams, index);
        int width = mLayerParams.outputShape[0];
        int height = mLayerParams.outputShape[1];
        int channel = mLayerParams.outputShape[2];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        float[] localInput = new float[width * height * 4];
        for (int i = 0; i < width * height; i++) {
            for (int ii = 0; ii < 4; ii++) {
                if (index * 4 + ii < channel) {
                    localInput[i * 4 + ii] = input[index * 4 + ii][i];
                } else {
                    localInput[i * 4 + ii] = 0;

                }
            }
        }
        ComputeRender.transferToTexture(FloatBuffer.wrap(localInput), texID, startX, startY, width, height);
    }

}
