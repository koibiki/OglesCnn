package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

public class PoolingLayer extends Layer {

    private final int[] mKsize;
    private final int[] mStrides;
    private Layer mPreLayer;
    private int mNumGroupsY;
    private int mShaderPro;
    private int[] mParams;


    public PoolingLayer(Context context, Layer preLayer, int[] shape, int[] ksize, int[] strides) {
        super(context, shape);
        this.mPreLayer = preLayer;
        this.mKsize = ksize;
        this.mStrides = strides;
    }

    private void initPooling() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "pooling.comp", mKsize[0] * mKsize[1], mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);

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
    protected void actualForwardProc() {
        ComputeRender.performWithIntParams(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
