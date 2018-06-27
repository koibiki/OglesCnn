package com.example.cnnlib.layer;

import android.content.Context;

import com.example.cnnlib.render.ComputeRender;
import com.example.cnnlib.utils.AttachIDManager;
import com.example.cnnlib.utils.DataUtils;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.ComputeRender.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.ComputeRender.initConvolute3Pro;

/**
 * 使用 ssbo 存储kennel
 * 每个计算器同时计算出输出的4个通道上的数据  17.9 ~ 19.1
 */
public class ConvolutionLayer4 extends Layer {

    private static final String TAG = "ConvolutionLayer3";

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private NonLinearLayer.NonLinearType mType;

    private int[] mKennelBuffer = new int[1];


    public ConvolutionLayer4(Context context, Layer preLayer, int[] shape, int[] kennelShape, int padding, int[] strides, NonLinearLayer.NonLinearType type) {
        super(context, shape, preLayer);
        this.mKennelShape = kennelShape;
        this.mPadding = padding;
        this.mStrides = strides;
        this.mType = type;
    }

    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        int localSizeZ = getCompShaderLocalSizeZ(mOutputShape, 4);
        mNumGroupsZ = (int) Math.ceil(mOutputShape[2] * 1.0d / (localSizeZ * 4));

        mShaderPro = initConvolute3Pro(mContext, "conv4.comp", mKennelShape, mOutputShape[2], mOutputShape[0], localSizeY, localSizeZ);
        mAttachID = AttachIDManager.getInstance().getDataAttachID();
        mOutTex = ComputeRender.createTexture();

        int kennelBufSize = (mKennelShape[0] * mKennelShape[1] * mKennelShape[2] + 1) * mOutputShape[2];
        mKennelBuffer[0] = ComputeRender.initKennelBuffer(kennelBufSize);
        mKennels = createKennels();
        transferKennelToBuffer();

        mParams = new int[13];
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
            kennels.add(DataUtils.createConvKennel(i + 1, 3, 3, 4, 1));
        }
        return kennels;
    }


    private void transferKennelToBuffer() {
        for (int i = 0; i < mKennels.size(); i++) {
            int offset = i * (mKennelShape[0] * mKennelShape[1] * mKennelShape[2] + 1);
            ComputeRender.transferToBuffer(FloatBuffer.wrap(mKennels.get(i)), mKennelBuffer[0], offset);
        }
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        ComputeRender.bindTextureAndBuffer(mOutTex, mAttachID, mKennelBuffer[0]);
    }

    @Override
    protected void actualForwardProc() {
        ComputeRender.performConvolute3(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelBuffer[0], mNumGroupsY, mNumGroupsZ);
    }

}
