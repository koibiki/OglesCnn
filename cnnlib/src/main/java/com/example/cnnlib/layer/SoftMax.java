package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.Render;

import static com.example.cnnlib.render.Render.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.Render.initSoftPro;

public class SoftMax extends Layer {

    private static final String TAG = "SoftMax";

    private int mShaderPro;
    private int mNumGroupsY;
    private int mAmount;

    public SoftMax(Context context, Layer preLayer) {
        super(context, preLayer);
        this.mAmount = preLayer.getOutputShape()[2];
        this.mOutputShape = preLayer.getOutputShape();
    }

    private void initSoftmax() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initSoftPro(mContext, "softmax.comp", mAmount, mOutputShape[0], localSizeY, 1);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();
    }

    @Override
    public void initialize() {
        initSoftmax();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performWithoutParams(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
