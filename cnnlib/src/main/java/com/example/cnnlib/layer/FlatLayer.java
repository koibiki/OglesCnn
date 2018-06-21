package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.initCompPro;

/**
 * 输出维度为输入的 w*h, c, 1
 */
public class FlatLayer extends Layer {

    private Layer mPreLayer;
    private int mShaderPro;
    private int[] mParams;

    public FlatLayer(Context context, Layer preLayer, int[] shape) {
        super(context, shape);
        this.mPreLayer = preLayer;
    }

    private void initFlat() {
        mShaderPro = initCompPro(mContext, "flat.comp", mOutputShape[0], 1);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mOutputShape[0], mOutputShape[1]);

        int[] inputShape = mPreLayer.getOutputShape();
        mParams = new int[10];
        mParams[0] = inputShape[0];
        mParams[1] = inputShape[1];
        mParams[2] = inputShape[2];
        mParams[3] = mOutputShape[0];
        mParams[4] = mOutputShape[1];
        mParams[5] = mOutputShape[2];
    }

    @Override
    public void initialize() {
        initFlat();
    }


    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performWithIntParams(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, 1);
    }

}
