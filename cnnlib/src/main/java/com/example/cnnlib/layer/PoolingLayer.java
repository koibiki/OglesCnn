package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

public class PoolingLayer extends Layer {

    private int mNumGroupsY;
    private int mShaderPro;
    private int[] mParams;


    public PoolingLayer(Context context, LayerParams layerParams, int[] ksize, int[] strides) {
        super(context, layerParams);
        initPooling(ksize, strides);
    }

    private void initPooling(int[] ksize, int[] strides) {
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "pooling.comp", ksize[0] * ksize[1], mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);

        mParams = new int[10];
        mParams[0] = mLayerParams.inputShape[0];
        mParams[1] = mLayerParams.inputShape[1];
        mParams[2] = mLayerParams.inputShape[2];
        mParams[3] = mLayerParams.outputShape[0];
        mParams[4] = mLayerParams.outputShape[1];
        mParams[5] = mLayerParams.outputShape[2];
        mParams[6] = ksize[0];
        mParams[7] = ksize[1];
        mParams[8] = strides[0];
        mParams[9] = strides[1];
    }

    @Override
    public int forwardProc(int inTex) {
        ComputeRender.performWithIntParams(mShaderPro, mParams, inTex, mOutTex, mNumGroupsY);
        return mOutTex;
    }

    @Override
    public void readOutput() {

    }
}
