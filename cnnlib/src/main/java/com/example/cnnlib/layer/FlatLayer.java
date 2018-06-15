package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.initCompPro;

/**
 * 输入限制 尺寸 1024 X 1024 X 1
 */
public class FlatLayer extends Layer {

    private int mShaderPro;
    private int[] mParams;

    public FlatLayer(Context context, LayerParams layerParams) {
        super(context, layerParams);
        initFlat();
    }

    private void initFlat() {
        mShaderPro = initCompPro(mContext, "flat.comp", mLayerParams.outputShape[0], 1);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);

        mParams = new int[10];
        mParams[0] = mLayerParams.inputShape[0];
        mParams[1] = mLayerParams.inputShape[1];
        mParams[2] = mLayerParams.inputShape[2];
        mParams[3] = mLayerParams.outputShape[0];
        mParams[4] = mLayerParams.outputShape[1];
        mParams[5] = mLayerParams.outputShape[2];
    }

    @Override
    public int forwardProc(int inTex) {
        ComputeRender.performWithIntParams(mShaderPro, mParams, inTex, mOutTex, 1);
        return mOutTex;
    }

    @Override
    public void readOutput(){

    }

}
