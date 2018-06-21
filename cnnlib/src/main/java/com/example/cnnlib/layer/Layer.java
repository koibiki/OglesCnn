package com.example.cnnlib.layer;

import android.content.Context;

public abstract class Layer {

    protected Context mContext;
    protected int[] mOutputShape;

    protected int mOutTex;
    protected int mAttachID;

    public Layer(Context context, int[] shape) {
        this.mContext = context;
        this.mOutputShape = shape;
    }

    public int getAttachID() {
        return mAttachID;
    }

    public int getOutTex() {
        return mOutTex;
    }

    public int[] getOutputShape() {
        return mOutputShape;
    }

    public abstract void initialize();

    protected abstract void bindTextureAndBuffer();

    protected abstract void actualForwardProc();

    public void forwardProc() {
        bindTextureAndBuffer();
        actualForwardProc();
    }

    public abstract void readOutput();

}
