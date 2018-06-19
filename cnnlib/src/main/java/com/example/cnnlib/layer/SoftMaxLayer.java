package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initCompPro;
import static com.example.cnnlib.render.ComputeRender.initPoolingPro;

/**
 * softmax 种类不能超过 1024 前接全连接
 */
public class SoftMaxLayer extends Layer {

    private static final String TAG = "SoftMaxLayer";
    private int mShaderPro;
    private int mNumGroupsY;

    public SoftMaxLayer(Context context, LayerParams layerParams) {
        super(context, layerParams);
        initSoftmax();
    }

    private void initSoftmax() {
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initPoolingPro(mContext, "softmax.comp", mLayerParams.outputShape[0] * mLayerParams.outputShape[1], mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);
    }

    @Override
    public int forwardProc(int inTex) {
        ComputeRender.performWithoutParams(mShaderPro, inTex, mOutTex, mNumGroupsY);
        return mOutTex;
    }

    @Override
    public void readOutput() {

    }


}
