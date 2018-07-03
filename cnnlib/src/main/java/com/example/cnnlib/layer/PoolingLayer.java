package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

public class PoolingLayer extends Layer {

    private int[] mKsize;
    private int[] mStrides;
    private int mNumGroupsY;
    private int mShaderPro;
    private int[] mParams;
    private int mNumGroupsZ;


    public PoolingLayer(Context context, Layer preLayer, int[] ksize, int[] strides) {
        super(context, preLayer);
        this.mKsize = ksize;
        this.mStrides = strides;
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
        mOutTex = ComputeRender.createTexture();

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
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        ComputeRender.performWithIntParams(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsY, mNumGroupsZ);
    }

}
