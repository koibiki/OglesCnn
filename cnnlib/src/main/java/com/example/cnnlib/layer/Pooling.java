package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.Render;

import static com.example.cnnlib.render.Render.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.Render.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.Render.initPoolingPro;

public class Pooling extends Layer {

    private int[] mKsize;
    private int[] mStrides;
    private int mNumGroupsY;
    private int mShaderPro;
    private int[] mParams;
    private int mNumGroupsZ;


    public Pooling(Context context, Layer preLayer, int k_w, int k_h, int stride_w, int stride_h) {
        super(context, preLayer);
        this.mKsize = new int[]{k_w, k_h};
        this.mStrides = new int[]{stride_w, stride_h};
        this.mOutputShape = calculatePoolingShape();
    }

    private int[] calculatePoolingShape() {
        int[] inShape = mPreLayer.getOutputShape();
        int width = (int) Math.ceil(inShape[0] * 1.0f / mStrides[0]);
        int height = (int) Math.ceil(inShape[1] * 1.0f / mStrides[1]);
        return new int[]{width, height, inShape[2]};
    }

    private void initPooling() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        int localSizeZ = getCompShaderLocalSizeZ(mOutputShape, 4);
        mNumGroupsZ = (int) Math.ceil(mOutputShape[2] * 1.0d / (localSizeZ * 4));

        mShaderPro = initPoolingPro(mContext, "pooling.comp", mKsize[0] * mKsize[1], mOutputShape[0], localSizeY, localSizeZ);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();

        int[] inputShape = mPreLayer.getOutputShape();
        mParams = new int[10];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
        mParams[6] = mKsize[0];
        mParams[7] = mKsize[1];
        mParams[8] = mStrides[0];
        mParams[9] = mStrides[1];
    }

    @Override
    public void initialize() {
        initPooling();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performWithIntParams(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsY, mNumGroupsZ);
    }

}
