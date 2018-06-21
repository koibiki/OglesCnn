package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUtils;

import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.initConvolutePro;

public class ConvolutionLayer extends Layer {

    private static final String TAG = "ConvolutionLayer";

    private Layer mPreLayer;
    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int[] mParams;
    private NonLinearLayer.NonLinearType mType;


    public ConvolutionLayer(Context context, Layer preLayer, int[] shape, int[] kennelShape, int padding, int[] strides, NonLinearLayer.NonLinearType type) {
        super(context, shape);
        this.mPreLayer = preLayer;
        this.mKennelShape = kennelShape;
        this.mPadding = padding;
        this.mStrides = strides;
        this.mType = type;

    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        mShaderPro = initConvolutePro(mContext, "conv.comp", mKennelShape, mOutputShape[0], localSizeY);
        mAttachID = AttachIDManager.getInstance().getAttachID();
        mOutTex = ComputeRender.createTexture(mAttachID);

        mKennels = createKennels();

        mParams = new int[14];
        int[] inputShape = mPreLayer.getOutputShape();
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = inputShape[0];
        mParams[4] = inputShape[1];
        mParams[5] = inputShape[2];
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
        mParams[12] = mType.index;
    }

    private List<float[]> createKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, 3, 4, 1));
        }
        return kennels;
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID);
    }

    @Override
    protected void actualForwardProc() {
        for (int i = 0; i < mKennels.size(); i++) {
            mParams[13] = i;
            ComputeRender.performConvolute(mShaderPro, mKennels.get(i), mParams, mPreLayer.getOutTex(), mOutTex, mNumGroupsY);
        }
    }

}
