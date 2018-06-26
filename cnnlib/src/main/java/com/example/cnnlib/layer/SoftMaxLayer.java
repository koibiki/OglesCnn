package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

/**
 * softmax 种类不能超过 1024 前接全连接
 */
public class SoftMaxLayer extends Layer {

    private static final String TAG = "SoftMaxLayer";

    private int mShaderPro;
    private int mNumGroupsY;

    public SoftMaxLayer(Context context, Layer preLayer, int[] shape) {
        super(context, shape, preLayer);
    }

    private void initSoftmax() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "softmax.comp", mOutputShape[0] * mOutputShape[1], mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getDataAttachID();
        mOutTex = ComputeRender.createTexture();
    }

    @Override
    public void initialize() {
        initSoftmax();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performWithoutParams(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
