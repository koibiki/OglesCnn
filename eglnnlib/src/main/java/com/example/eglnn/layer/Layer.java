package com.example.eglnn.layer;

import android.content.Context;

public abstract class Layer {

    protected Context mContext;
    protected Layer mPreLayer;
    protected int[] mOutShape;
    protected int[] mInShape;

    protected int mOutTex;
    protected int mAttachID;

    private static int sStartY = 0;

    private static int mCurrentDataId = 6;

    public enum PaddingType {
        SAME(0), VALID(1);

        public int index;

        PaddingType(int index) {
            this.index = index;
        }
    }

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

    public Layer(Context context, int w, int h, int c) {
        this.mContext = context;
        this.mPreLayer = null;
        this.mInShape = new int[]{w, h, c};
    }


    public Layer(Context context, Layer preLayer) {
        this.mContext = context;
        this.mPreLayer = preLayer;
        this.mInShape = preLayer.getOutputShape();
    }


    public int getAttachID() {
        return mAttachID;
    }

    public int getOutTex() {
        return mOutTex;
    }

    public int[] getOutputShape() {
        return mOutShape;
    }

    public abstract void initialize();

    protected abstract void bindTextureAndBuffer();

    protected abstract void actualForwardProc(float[][] input);

    public void forwardProc(float[][] input) {
        bindTextureAndBuffer();
        actualForwardProc(input);
    }

}
