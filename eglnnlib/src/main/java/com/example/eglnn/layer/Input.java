package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;

public class Input extends Layer {

    static int i = 0;

    public Input(Context context, int w, int h, int c) {
        super(context, w, h, c);
        this.mOutShape = new int[]{w, h, c};
    }

    @Override
    public void initialize() {
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) / 4);
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
//        float[][][] input1 = TestDataCreator.createInputBuffer(new int[]{mOutShape[0], mOutShape[1], mOutShape[2]}, ++i);
//        float[][] localInput = new float[Utils.alignBy4(mOutShape[2]) / 4][mOutShape[0] * mOutShape[1] * 4];
//        for (int w = 0; w < mOutShape[0]; w++) {
//            for (int h = 0; h < mOutShape[1]; h++) {
//                for (int c = 0; c < mOutShape[2]; c++) {
//                    localInput[c / 4][(h * mOutShape[0] + w) * 4 + c % 4] = input1[c][h][w];
//                }
//            }
//        }
        writeInput(input);
    }

    private void writeInput(float[][] input) {
        for (int i = 0; i < input.length; i++) {
            FloatBuffer data = FloatBuffer.wrap(input[i]);
            Render.transferToTextureArrayFloat(data, mOutTex, 0, 0, i, mOutShape[0], mOutShape[1], 1);
        }
    }

}
