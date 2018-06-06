package com.example.cnnlib.layer;

import android.content.Context;
import android.util.Log;
import android.util.StateSet;

import com.example.cnnlib.model.Kennel;
import com.example.cnnlib.model.LayerParams;
import com.example.cnnlib.render.ComputeRender;

import java.util.List;

public class ConvolutionLayer extends Layer {

    private static final String TAG = "ConvolutionLayer";

    private List<Kennel> mKennels;
    private int[] mStrides;
    private int mPadding;
    private int mPaddingType;


    public ConvolutionLayer(Context context, LayerParams layerParams, List<Kennel> kennels, int padding, int paddingType, int[] strides) {
        super(context, layerParams);
        this.mKennels = kennels;
        this.mPadding = padding;
        this.mPaddingType = paddingType;
        this.mStrides = strides;
    }

    @Override
    public int forwardProc(int InputTexID, int attachID) {
        mOutputTexID = ComputeRender.createTexture(attachID);
        for (int i = 0; i < mKennels.size(); i++) {
            long begin = System.currentTimeMillis();
            ComputeRender.performConvolute(mContext, "conv.comp", InputTexID, mOutputTexID, mKennels.get(i), i, mLayerParams, mPadding, mPaddingType, mStrides);
            Log.w(TAG, "convolute:" + i + "spent:" + (System.currentTimeMillis() - begin));
        }
        return mOutputTexID;
    }


}
