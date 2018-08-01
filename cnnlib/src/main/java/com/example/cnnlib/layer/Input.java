package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.Constants;
import com.example.cnnlib.utils.NetUtils;

import java.nio.FloatBuffer;

public class Input extends Layer {

    public Input(Context context, int w, int h, int c) {
        super(context, null);
        this.mOutputShape = new int[]{w, h, c};
    }

    @Override
    public void initialize() {
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID);
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
                        localInput[(h * width + w) * 4 + ii] = input[index * 4 + ii][h][w];
                    } else {
                        localInput[(h * width + w) * 4 + ii] = 0;

                    }
                }
            }
        }
        Render.transferToTexture(FloatBuffer.wrap(localInput), texID, startX, startY, width, height);
    }

}
