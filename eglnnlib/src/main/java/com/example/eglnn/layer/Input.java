package com.example.eglnn.layer;

import android.content.Context;

import com.example.eglnn.Render;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;

public class Input extends Layer {

    public Input(Context context, String name, int w, int h, int c) {
        super(context, name, w, h, c);
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
        writeInput(input);
    }

    private void writeInput(float[][] input) {
        for (int i = 0; i < input.length; i++) {
            FloatBuffer data = FloatBuffer.wrap(input[i]);
            Render.transferToTextureArrayFloat(data, mOutTex, 0, 0, i, mOutShape[0], mOutShape[1], 1);
        }
    }

}
