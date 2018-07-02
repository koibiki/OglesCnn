package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.NetUtils;

import java.nio.FloatBuffer;

public class InputLayer extends Layer {

    public InputLayer(Context context, int[] shape) {
        super(context, shape, null);
    }

    @Override
    public void initialize() {
        mAttachID = Layer.getDataAttachID();
        mOutTex = ComputeRender.createTexture();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        int channel = mOutputShape[2];
        int count = NetUtils.alignBy4(channel);
        for (int i = 0; i < count / 4; i++) {
            writeInput(input, mOutTex, i);
        }
    }

    private void writeInput(float[][][] input, int texID, int index) {
        int width = mOutputShape[0];
        int[] indexes = NetUtils.getXYIndex(width, index, Constants.S_TEXTURE_SIZE);
        int height = mOutputShape[1];
        int channel = mOutputShape[2];
        int startX = indexes[0] * width;
        int startY = indexes[1] * height;
        float[] localInput = new float[width * height * 4];
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
                for (int ii = 0; ii < 4; ii++) {
                    if (index * 4 + ii < channel) {
                        localInput[(h * width + w) * 4 + ii] = input[index * 4 + ii][w][h];
                    } else {
                        localInput[(h * width + w) * 4 + ii] = 0;

                    }
                }
            }
        }
        ComputeRender.transferToTexture(FloatBuffer.wrap(localInput), texID, startX, startY, width, height);
    }

}
