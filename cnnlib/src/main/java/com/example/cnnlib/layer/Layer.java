package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.utils.DataUtils;

import java.util.List;

public abstract class Layer {

    protected Context mContext;
    protected int[] mOutputShape;

    protected int mOutTex;
    protected int mAttachID;
    private List<float[]> mResult;

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

    public void forwardProc(boolean restore) {
        bindTextureAndBuffer();
        actualForwardProc();
        if (restore) {
            restoreResult();
        }
    }

    private void restoreResult() {
        mResult = DataUtils.readOutput(this);
    }

}
