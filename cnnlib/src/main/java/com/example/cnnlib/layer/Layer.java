package com.example.cnnlib.layer;

import android.content.Context;

public abstract class Layer {

    protected Context mContext;
    protected Layer mPreLayer;
    protected int[] mOutputShape;

    protected int mOutTex;
    protected int mAttachID;

    private static int sStartY = 0;

    private static int mCurrentDataId = 6;

    protected static int getDataAttachID() {
        mCurrentDataId++;
        return mCurrentDataId % 7;
    }

    public static int getConvKennelAttachID() {
        return 7;
    }

    public static int getConvStartY(int ySize) {
        int startY = sStartY;
        sStartY += ySize;
        return startY;
    }


    public Layer(Context context, Layer preLayer) {
        this.mContext = context;
        this.mPreLayer = preLayer;
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

    protected abstract void actualForwardProc(float[][][] input);

    public void forwardProc(float[][][] input) {
        bindTextureAndBuffer();
        actualForwardProc(input);
    }

}
