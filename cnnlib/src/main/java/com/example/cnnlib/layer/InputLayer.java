package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.SortUtils;

import java.nio.FloatBuffer;

public class InputLayer extends Layer {

    public InputLayer(Context context, LayerParams layerParams) {
        super(context, layerParams);
        this.mAttachID = AttachIDManager.getInstance().getAttachID();
        this.mOutputTexID = ComputeRender.createTexture(mAttachID);
    }

    private float[][] createInputBuffer() {
        int[] inputShape = mLayerParams.inputShape;
        int channel = inputShape[2];
        int areaCapacity = inputShape[0] * inputShape[1];
        float input[][] = new float[channel][areaCapacity];
        for (int j = 0; j < channel; j++) {
            for (int i = 0; i < areaCapacity; i++) {
                int s = i%2==0? -1:1;
                input[j][i] = i % inputShape[0] * s;
            }
        }
        return input;
    }

    @Override
    public int forwardProc(int inputTexID) {
        float[][] input = createInputBuffer();

        int count = SortUtils.getCount(mLayerParams);
        for (int i = 0; i < count; i++) {
            writeInput(input, mOutputTexID, i);
        }
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
