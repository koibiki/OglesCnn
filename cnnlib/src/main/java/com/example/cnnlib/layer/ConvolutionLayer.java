package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;

import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initConvolutePro;

public class ConvolutionLayer extends Layer {

    private static final String TAG = "ConvolutionLayer";

    private List<Kennel> mKennels;
    private int[] mStrides;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int[] mParams;


    public ConvolutionLayer(Context context, LayerParams layerParams, List<Kennel> kennels, int padding, int[] strides) {
        super(context, layerParams);
        this.mKennels = kennels;
        this.mPadding = padding;
        this.mStrides = strides;

        initConv();
    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mLayerParams.outputShape);
        mNumGroupsY = (int) Math.ceil(mLayerParams.outputShape[1] * 1.0d / localSizeY);
        mShaderPro = initConvolutePro(mContext, "conv.comp", mKennels.get(0), mLayerParams.outputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);
        mParams = new int[13];

        mParams[0] = mKennels.get(0).shape[0];
        mParams[1] = mKennels.get(0).shape[1];
        mParams[2] = mKennels.get(0).shape[2];
        mParams[3] = mLayerParams.inputShape[0];
        mParams[4] = mLayerParams.inputShape[1];
        mParams[5] = mLayerParams.inputShape[2];
        mParams[6] = mLayerParams.outputShape[0];
        mParams[7] = mLayerParams.outputShape[1];
        mParams[8] = mLayerParams.outputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
    }


    @Override
    public int forwardProc(int inTex) {
        for (int i = 0; i < mKennels.size(); i++) {
            mParams[12] = i;
            ComputeRender.performConvolute(mShaderPro, mKennels.get(i), mParams, inTex, mOutTex, mNumGroupsY);
        }
        return mOutTex;
    }

    @Override
    public void readOutput() {

    }


}
