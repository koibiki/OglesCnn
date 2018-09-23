package com.example.eglnn.layer;

import android.content.Context;
import android.text.TextUtils;

import com.example.eglnn.Render;
import com.example.eglnn.utils.ShaderUtils;
import com.example.eglnn.utils.TestDataCreator;
import com.example.eglnn.utils.Utils;

import java.nio.FloatBuffer;
import java.util.Locale;

import static com.example.eglnn.Render.initCompPro;
import static com.example.eglnn.utils.Constants.S_CONV_GEMM_SHADER_HEADER;

/**
 *
 * */
public class ConvGEMM2 extends Layer {

    private static final String TAG = "ConvGEMM";
    private final int mKennelAmount;
    private PaddingType mPadType;

    private int mNumGroupsX;

    private float[][][] mKennels;
    private int[] mStrides;
    private int[] mKennelShape;
    private int mShaderPro;
    private int mNumGroupsZ;
    private int[] mParams;
    private ActiveType mType;
    private String mKennelFilePath;
    private int mKennelTex;
    private int mPadW, mPadH;

    private ConvGEMM2(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        super(context, preLayer);
        this.mKennelShape = new int[]{k_w, k_h, preLayer.getOutputShape()[2]};
        this.mKennelAmount = kAmount;
        this.mStrides = new int[]{stride_w, stride_h};
        this.mType = type;
        this.mKennelFilePath = kennelFilePath;
    }

    public ConvGEMM2(Context context, Layer preLayer, int kAmount, int k_w, int k_h, PaddingType padType, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        this(context, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type, kennelFilePath);
        this.mPadType = padType;
        this.mOutShape = calculateConvShapeByType(kAmount);
    }

    public ConvGEMM2(Context context, Layer preLayer, int kAmount, int k_w, int k_h, int pad_w, int pad_h, int stride_w, int stride_h, ActiveType type, String kennelFilePath) {
        this(context, preLayer, kAmount, k_w, k_h, stride_w, stride_h, type, kennelFilePath);
        this.mPadW = pad_w;
        this.mPadH = pad_h;
        this.mOutShape = calculateConvShapeByPad(kAmount);
    }

    private int[] calculateConvShapeByType(int kennelAmount) {
        int width, height;
        if (mPadType == PaddingType.SAME) {
            width = (int) Math.ceil(mInShape[0] * 1.0f / mStrides[0]);
            mPadW = ((width - 1) * mStrides[0] + mKennelShape[0] - mInShape[0]) / 2;
            height = (int) Math.ceil(mInShape[1] * 1.0f / mStrides[1]);
            mPadH = ((height - 1) * mStrides[1] + mKennelShape[1] - mInShape[1]) / 2;
        } else {
            width = (int) Math.ceil((mInShape[0] - mKennelShape[0] + 1) * 1.0f / mStrides[0]);
            height = (int) Math.ceil((mInShape[1] - mKennelShape[1] + 1) * 1.0f / mStrides[1]);
        }
        return new int[]{width, height, kennelAmount};
    }

    private int[] calculateConvShapeByPad(int kennelAmount) {
        int width = (int) Math.ceil((mInShape[0] + 2 * mPadW - mKennelShape[0]) * 1.0f / mStrides[0]);
        int height = (int) Math.ceil((mInShape[1] + 2 * mPadH - mKennelShape[1]) * 1.0f / mStrides[1]);
        return new int[]{width, height, kennelAmount};
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
        mOutTex = Render.createFloatTextureArray(mOutShape[0], mOutShape[1], Utils.alignBy4(mOutShape[2]) * 2 / 4);

        if (TextUtils.isEmpty(mKennelFilePath)) {
            mKennels = createTestKennels();
        } else {
            mKennels = loadKennels();
        }
        // kennel 最后一列为bias
        mKennelTex = Render.createKennelFloatTextureArray(mKennelShape[0] * mKennelShape[1] + 1, mKennelAmount, Utils.alignBy4(mInShape[2]) / 4);
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
        String shaderFile = "conv_gemm2.comp";
        String source = ShaderUtils.loadFromAssetsFile(shaderFile, mContext.getResources());
        int kennelArea = mKennelShape[0] * mKennelShape[1];
        int kennelSize = kennelArea * Utils.alignBy4(mKennelShape[2]);
        return String.format(Locale.getDefault(), S_CONV_GEMM_SHADER_HEADER, kennelArea, mKennelAmount, kennelSize, xSize, 1, zSize) + source;
    }

    private void createShaderParams() {
        mParams = new int[15];
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
        mParams[11] = mType.index;
        mParams[12] = Utils.alignBy4(mInShape[2]);
        mParams[13] = -1 * mPadW;
        mParams[14] = -1 * mPadH;
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
        Render.bindTextureArray(mOutTex, mAttachID, mOutShape[2]/4 -1);
    }

    @Override
    protected void actualForwardProc(float[][] input) {
        Render.performConvoluteGEMM2(mShaderPro, mParams, mPreLayer.getOutTex(), mOutTex, mKennelTex, mNumGroupsX, mNumGroupsZ);
    }
}
