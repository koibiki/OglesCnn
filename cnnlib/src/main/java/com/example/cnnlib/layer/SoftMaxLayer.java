package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initSoftPro;

public class SoftMaxLayer extends Layer {

    private static final String TAG = "SoftMaxLayer";

    private int mShaderPro;
    private int mNumGroupsY;
    private int mAmount;

    public SoftMaxLayer(Context context, Layer preLayer, int amount) {
        super(context, preLayer);
        this.mAmount = amount;
        this.mOutputShape = calculateSoftShape(amount);
    }

    private int[] calculateSoftShape(int amount) {
        int width = (int) Math.ceil(amount * 1.0f / 4);
        int height = 1;
        int channel = 4;
        return new int[]{width, height, channel};
    }

    private void initSoftmax() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initSoftPro(mContext, "softmax.comp", mAmount, mOutputShape[0], localSizeY, 1);
        mAttachID = Layer.getDataAttachID();
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
    protected void actualForwardProc(float[][][] input) {
        ComputeRender.performWithoutParams(mShaderPro, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
    }

}
