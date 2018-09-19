package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.Constants;
import com.example.eglnn.utils.DataUtils;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_CONV_GEMM_SHADER_HEADER;

/**
 * 输入, 输出  w h c 均需要 4 对齐
 */
public class ConvGEMM4 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;

    private int mNumGroupsX;

    private float[][][] mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mPad;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private int mType;
    private String mKennelFilePath;
    private int mConvTex;
    private int mKennelTex;

    // TODO VALID 未实现
    private PaddingType mPadType;

    public ConvGEMM4(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad, PaddingType padType, int stride_w, int stride_h, int type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mPad = pad;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mPadType = padType;
        this.mOutShape = calculateConvShape(kAmount);
        this.mKennelFilePath = kennelFilePath;
    }

    private int[] calculateConvShape(int kennelAmount) {
        int[] inShape = mPreLayer.getOutputShape();
        int width = calculateLength(inShape[0], mKennelShape[0], mStrides[0]);
        int height = calculateLength(inShape[1], mKennelShape[1], mStrides[1]);
        return new int[]{width, height, kennelAmount};
    }

    private int calculateLength(int length, int kennelLen, int stride) {
        float out = (length + 2 * mPad - kennelLen) * 1.0f / stride;
        if (mPadType == PaddingType.SAME) {
            return (int) (Math.ceil(out) + 1);
        } else {
            return (int) (Math.floor(out) + 1);
        }
    }

    private int getLocalSizeX(int[] outShape) {
        int outArea = outShape[0] * outShape[1];
        if (outArea >= 1024 * 4) {
            return 1024;
        } else {
            return (int) Math.ceil(outArea * 1.0f / 4);
        }
    }

    private int getLocalSizeZ(int xSize) {
        int maxZSize = 1024 / xSize >= 64 ? 64 : 1024 / xSize;
        int zSize = Utils.alignBy4(mOutShape[2]) / 4;
        if (zSize >= maxZSize) {
            return maxZSize;
        } else {
            return zSize;
        }
    }

    private void initConv() {
        int xSize = getLocalSizeX(mOutShape);
        mNumGroupsX = (int) Math.ceil(Utils.alignBy4(mOutShape[0] * mOutShape[1]) * 1.0f / 4 / xSize);

        int zSize = getLocalSizeZ(xSize);
        mNumGroupsZ = (int) Math.ceil(Utils.alignBy4(mOutShape[2]) * 1.0f / 4 / zSize);

        String source = createShaderSource(xSize, zSize);
        mShaderPro = initCompPro(source);
        mAttachID = Layer.getDataAttachID();
        mOutTex = Render.createFloatTextureArray(mOutShape[0] + 1, mOutShape[1] + 1, Utils.alignBy4(mOutShape[2]) / 4);
        createConvInputDataIndex();

        mKennelTex = Render.createKennelFloatTextureArray(mKennelShape[0] * mKennelShape[1] + 1, mKennelAmount, Utils.alignBy4(mInShape[2]) / 4);


        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }

        transferToKennelTex();

        createShaderParams();
    }

    private void transferToKennelTex() {
        for (int a = 0; a < mKennelAmount; a++) {
            float[][] kennel = mKennels[a];
            for (int c = 0; c < kennel.length; c++) {
                Render.transferToTextureArrayFloat(FloatBuffer.wrap(kennel[c]), mKennelTex, 0, a, c, mKennelShape[0] * mKennelShape[1] + 1, 1, 1);
            }
        }
    }

    private String createShaderSource(int xSize, int zSize) {
        String source = ShaderUtils.loadFromAssetsFile("conv_gemm4.comp", mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        int kennelSize = kennelArea * Utils.alignBy4(mKennelShape[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }

    private void createConvInputDataIndex() {
        int outArea = mOutShape[0] * mOutShape[1];
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        mConvTex = Render.createIntTextureArray(outArea, 1, kennelArea);
        int[][] convIndexes = new int[kennelArea][outArea * 4];
        int zeroIndexX = mInShape[0];
        int zeroIndexY = mInShape[1];

        for (int outX = 0; outX < mOutShape[0]; outX++) {
            for (int outY = 0; outY < mOutShape[1]; outY++) {
                for (int kX = 0; kX < mKennelShape[0]; kX++) {
                    for (int kY = 0; kY < mKennelShape[1]; kY++) {
                        int outIndex = outY * mOutShape[0] + outX;
                        int kIndex = kY * mKennelShape[0] + kX;
                        int indexXOnFeatureMap = -mPad + kX + mStrides[0] * outX;
                        int indexYOnFeatureMap = -mPad + kY + mStrides[1] * outY;
                        if (isZero(indexXOnFeatureMap, indexYOnFeatureMap)) {
                            convIndexes[kIndex][4 * outIndex] = zeroIndexX;
                            convIndexes[kIndex][4 * outIndex + 1] = zeroIndexY;
                            convIndexes[kIndex][4 * outIndex + 2] = zeroIndexX;
                            convIndexes[kIndex][4 * outIndex + 3] = zeroIndexY;
                        } else {
                            convIndexes[kIndex][4 * outIndex] = indexXOnFeatureMap;
                            convIndexes[kIndex][4 * outIndex + 1] = indexYOnFeatureMap;
                            convIndexes[kIndex][4 * outIndex + 2] = zeroIndexX;
                            convIndexes[kIndex][4 * outIndex + 3] = zeroIndexY;
                        }
                    }
                }
            }
        }
        for (int i = 0; i < kennelArea; i++) {
            Render.trainferToTexttureArrayInt(IntBuffer.wrap(convIndexes[i]), mConvTex, 0, 0, i, outArea, 1);
        }
    }

    private boolean isZero(int x, int y) {
        return (x < 0 || x >= mInShape[0]) || (y < 0 || y >= mInShape[1]);
    }

    private void createShaderParams() {
        mParams = new int[14];
        mParams[0] = mKennelShape[0];
        mParams[1] = mKennelShape[1];
        mParams[2] = mKennelShape[2];
        mParams[3] = mInShape[0];
        mParams[4] = mInShape[1];
        mParams[5] = mInShape[2];
        mParams[6] = mOutShape[0];
        mParams[7] = mOutShape[1];
        mParams[8] = mOutShape[2];
        mParams[9] = mStrides[0];
        mParams[10] = mStrides[1];
        mParams[11] = mPad;
        mParams[12] = mType;
        mParams[13] = Utils.alignBy4(mInShape[2]);
    }

    private float[][][] createTestKennels() {
        return TestDataCreator.createConvKennels(mKennelShape, mKennelAmount);
    }

    /**
     * TODO
     * channel需要4对齐, 依次排满一个通道后再排下一个通道的值, 最后4个值为 bias, 0, 0, 0
     */
    private float[][][] loadKennels() {
        return null;
    }


    @Override
    public void initialize() {
        initConv();
    }

    @Override
    protected void bindTextureAndBuffer() {
        Render.bindTextureArray(mOutTex, mAttachID, 0);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvoluteGEMM4(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mConvTex, mKennelTex, mNumGroupsX, mNumGroupsZ);
    }
}
