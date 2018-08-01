package com.example.cnnlib.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.cnnlib.render.Render;
import com.example.cnnlib.utils.DataUtils;
import com.example.cnnlib.utils.NetUtils;
import com.example.cnnlib.utils.ParamUnpacker;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.example.cnnlib.render.Render.getCompShaderLocalSizeY;
import static com.example.cnnlib.render.Render.getCompShaderLocalSizeZ;
import static com.example.cnnlib.render.Render.initConvolutePro;

/**
 * 使用 ssbo 存储kennel
 * 每个计算器同时计算出输出的4个通道上的数据  17.9 ~ 19.1
 * 注意：输入,输出,kennel的通道都必须4对齐
 */
public class ConvSSBO extends Layer {

    private static final String TAG = "ConvSSBO";

    private List<float[]> mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPadding;
    private int mShaderPro;
    private int mNumGroupsY;
    private int mNumGroupsZ;
    private int[] mParams;
    private NonLinear.NonLinearType mType;
    private String mKennelFilePath;

    private int[] mKennelBuffer = new int[1];


    public ConvSSBO(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, int stride_w, int stride_h, NonLinear.NonLinearType type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mPadding = pad;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mOutputShape = calculateConvShape(kAmount);
        this.mKennelFilePath = kennelFilePath;
    }

    private int[] calculateConvShape(int kennelAmount) {
        int[] inShape = mPreLayer.getOutputShape();
        int width = calculateLength(inShape[0], mKennelShape[0], mStrides[0]);
        int height = calculateLength(inShape[1], mKennelShape[1], mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int calculateLength(int length, int kennelLen, int stride) {
        return (length + 2 * mPadding - kennelLen) / stride + 1;
    }


    private void initConv() {
        int localSizeY = getCompShaderLocalSizeY(mOutputShape);
        mNumGroupsY = (int) Math.ceil(mOutputShape[1] * 1.0d / localSizeY);
        int localSizeZ = getCompShaderLocalSizeZ(mOutputShape, 4);
        mNumGroupsZ = (int) Math.ceil(mOutputShape[2] * 1.0d / (localSizeZ * 4));

        mShaderPro = initConvolutePro(mContext, "conv.comp", mKennelShape, mOutputShape[2], mOutputShape[0], localSizeY, localSizeZ);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createTexture();

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }
        int kennelBufSize = mKennels.get(0).length * mOutputShape[2];
        mKennelBuffer[0] = Render.initKennelBuffer(kennelBufSize);
        transferKennelToBuffer();

        mParams = new int[13];
        int[] inputShape = mPreLayer.getOutputShape();
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = NetUtils.alignBy4(mKennelShape[2]);
        mParams[3] = inputShape[0];
        mParams[4] = inputShape[1];
        mParams[5] = NetUtils.alignBy4(inputShape[2]);
        mParams[6] = mOutputShape[0];
        mParams[7] = mOutputShape[1];
        mParams[8] = mOutputShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPadding;
        mParams[12] = mType.index;
    }

    private List<float[]> createTestKennels() {
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            kennels.add(DataUtils.createConvKennel(i + 1, mKennelShape[0], mKennelShape[1], mKennelShape[2], 1));
        }
        return kennels;
    }

    /**
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private List<float[]> loadKennels() {
        ParamUnpacker paramUnpacker = new ParamUnpacker();
        Object[] objects = paramUnpacker.unpackerFunction(mKennelFilePath, new Class[]{float[][][][].class, float[].class});
        float[][][][] localWeight = (float[][][][]) objects[0];
        float[] localBias = (float[]) objects[1];

        int alignChannel = NetUtils.alignBy4(mKennelShape[2]);
        List<float[]> kennels = new ArrayList<>();
        for (int i = 0; i < mOutputShape[2]; i++) {
            float[] kennel = new float[mKennelShape[0] * mKennelShape[1] * alignChannel + 4];
            for (int c = 0; c < mKennelShape[2]; c++) {
                for (int w = 0; w < mKennelShape[0]; w++) {
                    for (int h = 0; h < mKennelShape[1]; h++) {
                        kennel[(h * mKennelShape[0] + w) * alignChannel + c] = localWeight[i][c][h][w];
                    }
                }
            }
            kennel[mKennelShape[0] * mKennelShape[1] * alignChannel] = localBias[i];
            kennels.add(kennel);
        }
        return kennels;
    }


    private void transferKennelToBuffer() {
        for (int i = 0; i < mKennels.size(); i++) {
            float[] kennel = mKennels.get(i);
            int offset = i * kennel.length;
            Render.transferToBuffer(FloatBuffer.wrap(kennel), mKennelBuffer[0], offset);
        }
    }

    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureAndBuffer(mOutTex, mAttachID, mKennelBuffer[0]);
    }

    @Override
    protected void actualForwardProc(float[][][] input) {
        Render.performConvolute(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelBuffer[0], mNumGroupsY, mNumGroupsZ);
    }

}
